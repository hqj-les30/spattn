import torch
from collections import deque
from .base import BaseServer, ServerSolver, register_server
from utils import compute_macro_f1, compute_accuracy, compute_output_dist, entropy
from data_wrapper import proxy_data_preparation
from agent import AgentSolver, Agent, AgentVec, make_agent
from flops import count_forward_macs, macs_to_flops, uplink_projected_bytes

@register_server('STCS')
class Server(BaseServer):
    def __init__(self, server_solver: ServerSolver):
        super().__init__(server_solver)
        for client in self.client_vec:
            client.attach_to_server(self)

        self.agent_solver = AgentSolver.from_args(self.solver.args)
        self.agent = make_agent(self.agent_solver)
        self.proxy_dataset = proxy_data_preparation(size=1000, dataset=self.solver.args.dataset)
        self.history = deque(maxlen=self.agent_solver.recall)
        self.slide_window = deque(maxlen=self.agent_solver.recall + 1)

        # combo (WT4): per-class EMA baseline for progress-relative reward (default off)
        self._prog_baseline = torch.zeros(self.num_classes, requires_grad=False)
        self.progress_relative_reward = getattr(self.solver.args, 'progress_relative_reward', False)
        self.reward_smooth = getattr(self.solver.args, 'reward_smooth', 0.05)

    def run(self):
        super().run()
        try:
            reward_ema = torch.zeros(self.num_classes, requires_grad=False)
            meta_info = {
                'num_clients': self.solver.n_clients,
                'cluster_size': self.solver.cluster_size,
                'selection_size': self.solver.selection_size,
                'global_epochs': self.solver.global_epoch
            }
            meta_info['efficiency'] = self.compute_efficiency_stats()
            self.save_details_to_jsonl(meta_info, mode='w')
            for epoch in range(self.solver.global_epoch):
                self.logger.info(f"Global Epoch {epoch} starts.")
                current_cluster = next(self.client_sampler)
                self.logger.info(f"Clients in current cluster: {current_cluster}")
                if len(current_cluster) <= 1:
                    self.logger.info(f"Skip this cluster")
                    eval_dict = {
                        'epoch': epoch,
                        'cluster': current_cluster,
                        'action': 'skip'
                    }
                    eval_dict.update(self._evaluate_global_model())
                    self.save_details_to_jsonl(eval_dict, mode='a')
                    continue
                client_update_results = self._run_local_update(current_cluster)
                output_dist = self._proxy_data_output_dist()
                state = self.agent.dicts2state(epoch, client_update_results, self.glob_model.state_dict(), output_dist, self.history)
                if len(current_cluster) >= self.solver.selection_size:
                    action, epsilon, behavior = self.agent.act(state, epoch)
                    selected_clients = [client_update_results[i] for i in action if i < len(client_update_results)]
                    self.logger.info(f"Action taken: {action};")
                else:
                    action = torch.arange(0, len(current_cluster))
                    epsilon = 0.0
                    behavior = 'takeall'
                    self.logger.info(f"Action taken: {action};")
                    selected_clients = [client_update_results[i] for i in action if i < len(client_update_results)]
                new_weights = self._aggregate_parameters(model_dicts=[client['model_params'] for client in selected_clients],
                                weights=[client['num_samples'] for client in selected_clients], model_fn=self.solver.tar_model_fn)
                if getattr(self.solver.args, 'no_temporal_agg', False):
                    # A1c: w/o temporal parameter aggregation -> standard FedAvg
                    self.glob_model.load_state_dict(new_weights)
                else:
                    self.slide_window.append(new_weights)
                    avg_weights = self._aggregate_parameters(model_dicts=self.slide_window,
                                    weights=[1] * len(self.slide_window), model_fn=self.solver.tar_model_fn)
                    self.glob_model.load_state_dict(avg_weights)
                score_current, _ = self._proxy_data_reward()
                metric_mean = score_current.mean()
                if self.progress_relative_reward:
                    # WT4: reward = score - EMA(score), removing the monotonic epoch trend so
                    # the Q-net cannot explain reward variance by training progress alone.
                    # Per-class detrend keeps the per-class structure the vectorized Q (WT5) needs.
                    reward_signal = score_current - self._prog_baseline
                    self._prog_baseline = self.reward_smooth * score_current + (1.0 - self.reward_smooth) * self._prog_baseline
                else:
                    # master: EMA-smoothed per-class reward (0.05*old + 0.95*new, no clone)
                    reward_ema = 0.05 * reward_ema + 0.95 * score_current
                    reward_signal = reward_ema
                self.agent.remember(state, action, reward_signal)
                tdloss, celoss, q_batch = self.agent.learn()
                self.history.append((state[2], torch.zeros(self.solver.selection_size)))

                self.logger.info(f"Selected clients: {[client['id'] for client in selected_clients]}, Metric: {metric_mean}, reward: {score_current}")
                eval_dict = {
                    'epoch': epoch,
                    'cluster': current_cluster,
                    'action': action.tolist(),
                    'selected': [client['id'] for client in selected_clients],
                    'epsilon': round(epsilon, 3),
                    'behavior': behavior,
                    'reward': score_current.round(decimals=2).tolist(),
                    'metric': round(metric_mean.item(), 3),
                    'entropy': round(entropy(output_dist).item(), 3),
                    'Q': q_batch,
                    'tdloss': tdloss,
                    'celoss': celoss
                }
                eval_dict.update(self._evaluate_global_model())
                self.save_details_to_jsonl(eval_dict, mode='a')
                self.logger.info(f"Global Epoch {epoch} ends.")
        finally:
            self._close_pool()

    def _proxy_data_output_dist(self):
        return compute_output_dist(
            self.glob_model,
            self.proxy_dataset,
            self.solver.gpus[hash(self.solver.log_path) % self.num_gpus],
            num_classes=self.num_classes
        )

    def _proxy_data_reward(self):
        if self.solver.metric == 'acc':
            return compute_accuracy(
                self.glob_model,
                self.proxy_dataset,
                self.solver.gpus[hash(self.solver.log_path) % self.num_gpus],
                num_classes=self.num_classes
            )
        else:
            return compute_macro_f1(
                self.glob_model,
                self.proxy_dataset,
                self.solver.gpus[hash(self.solver.log_path) % self.num_gpus],
                num_classes=self.num_classes
            )

    def _qnet_forward_macs(self) -> int:
        """One Q-net forward MACs at batch=1 (MACs scale linearly with batch for fixed N/recall)."""
        feature_dim = self.agent_solver.feature_dim
        recall = self.agent_solver.recall
        N = self.agent_solver.cluster_size
        sel = self.agent_solver.selection_size
        nclasses = self.agent_solver.action_dim
        device = self.agent_solver.device
        B = 1
        x_u = torch.zeros(B, N, feature_dim, device=device)
        x_s = torch.zeros(B, feature_dim, device=device)
        indicators = torch.zeros(B, N, dtype=torch.long, device=device)
        distribution = torch.zeros(B, nclasses, device=device)
        history = torch.zeros(B, recall, feature_dim + sel, device=device)
        try:
            return count_forward_macs(self.agent.qnetwork_local, (x_u, x_s, indicators, distribution, history, None))
        except Exception as e:
            self.logger.warning(f"Q-net MAC counting failed: {e}")
            return 0

    def compute_efficiency_stats(self):
        """Extend shared stats with STCS-specific RL server FLOPs and projected uplink."""
        stats = super().compute_efficiency_stats()
        feature_dim = self.agent_solver.feature_dim
        recall = self.agent_solver.recall
        selection_size = self.solver.selection_size
        cluster_size = self.solver.cluster_size
        full_param_size = self.full_param_size
        no_temporal_agg = getattr(self.solver.args, 'no_temporal_agg', False)
        encoder = getattr(self.agent_solver, 'encoder', 'attention')

        qnet_macs = self._qnet_forward_macs()
        b = self.agent_solver.b
        agent_batch = self.agent_solver.batch_size
        cnn_macs = stats['client_cnn_forward_macs']
        proxy_size = len(self.proxy_dataset)

        # server RL machinery: 1 inference (act) + b training batches/round.
        # per training batch: target fwd(next) + local fwd(next) + local fwd(cur) + backward(cur, ~2x fwd) = 5 forwards.
        act_flops = macs_to_flops(qnet_macs)
        train_flops = macs_to_flops(5 * b * agent_batch * qnet_macs)
        rl_flops = act_flops + train_flops

        # aggregation: weighted sum over selected (+ temporal-agg window unless A1c)
        agg_models = selection_size + (0 if no_temporal_agg else (recall + 1))
        agg_flops = macs_to_flops(full_param_size * agg_models)

        # proxy evaluation: output_dist + reward, each a full pass over the proxy set
        proxy_flops = macs_to_flops(cnn_macs * proxy_size * 2)

        uplink_proj = uplink_projected_bytes(cluster_size, feature_dim)
        uplink_full = stats['uplink_full_bytes']

        stats.update({
            'feature_dim': feature_dim,
            'recall': recall,
            'encoder': encoder,
            'no_temporal_agg': no_temporal_agg,
            'proxy_size': proxy_size,
            'uplink_projected_bytes': uplink_proj,
            'uplink_total_bytes': uplink_proj + uplink_full,
            'server_flops': {
                'rl_flops': int(rl_flops),
                'aggregation_flops': int(agg_flops),
                'proxy_eval_flops': int(proxy_flops),
                'total': int(rl_flops + agg_flops + proxy_flops),
                'qnet_forward_macs_b1': int(qnet_macs),
                'agent_batch_size': agent_batch,
                'agent_b': b,
            },
        })
        return stats
