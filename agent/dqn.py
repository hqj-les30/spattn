import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from models import set_qnet_fn, set_model_fn
from dataclasses import dataclass
from utils import parse_modelfunc, parse_devices, linear_decay, exponential_decay, MetaInfoQueue, entropy
from utils import project_X_onto_A_column_space as proj
from typing import Callable, Tuple, Type, Dict, List
import logging
from featuredim import set_feature_reducer, BaseClusterReducer
from torch.nn.utils.rnn import pad_sequence

class MultiStepReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99, device="cpu", padding = False):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
        self.padding = padding
        self.nstep_buffer = deque(maxlen=n_step + 1)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state"]
        )

    def __len__(self):
        return len(self.buffer)

    def _pad_varlen(self, tokens: List):
        lengths = torch.tensor([x.size(0) for x in tokens])
        padded = pad_sequence(tokens, batch_first=True)
        B, Lmax, _ = padded.shape
        mask = torch.arange(Lmax).expand(B, Lmax) >= lengths.unsqueeze(1)
        return padded, mask

    def add(self, state, action, reward):
        self.nstep_buffer.append((state, action, reward))
        if len(self.nstep_buffer) < self.n_step + 1:
            return
        (s1, s2, s3, s4, s5), a0, _ = self.nstep_buffer[0]
        (ns1, ns2, ns3, ns4, ns5), _, _ = self.nstep_buffer[-1]
        R = self._get_nstep_return()

        self.buffer.append(
            self.experience(
                (s1, s2, s3, s4, s5),
                a0, R,
                (ns1, ns2, ns3, ns4, ns5),
            )
        )

    def _repack_state(self, statelist: Tuple):
        client_ids, flat_params, flat_glob_param, output_dist, history_tensor = tuple(zip(*statelist))
        client_ids = pad_sequence(client_ids, batch_first=True)
        flat_params, mask = self._pad_varlen(flat_params)
        states = (
            client_ids, flat_params, torch.stack(flat_glob_param, dim=0),
            torch.stack(output_dist, dim=0), torch.stack(history_tensor, dim=0), mask
        )
        return states

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        rewards = torch.stack([e.reward for e in experiences], dim=0)
        if self.padding:
            states = self._repack_state([e.state for e in experiences])
            next_states = self._repack_state([e.next_state for e in experiences])
            actions = pad_sequence([e.action for e in experiences], batch_first=True)
        else:
            states = tuple([
                torch.stack(x, dim=0) for x in zip(*[e.state for e in experiences])
                ])
            next_states = tuple([
                torch.stack(x, dim=0) for x in zip(*[e.next_state for e in experiences])
                ])
            actions = torch.stack([e.action for e in experiences], dim=0)
        return (states, actions, rewards, next_states)

    def _get_nstep_return(self):
        R = torch.zeros_like(self.nstep_buffer[0][2])
        for i, c in enumerate(self.nstep_buffer):
            _, _, r = c
            if i == len(self.nstep_buffer) -1:
                break
            R += (self.gamma ** i) * r
        return R

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state"]
        )

    def add(self, state, action, reward, next_state):
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = tuple([
            torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
            for x in zip(*[e.state for e in experiences])
            ])
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.int64)
        rewards = tuple([
            torch.stack(x, dim=0) for x in zip(*[e.reward for e in experiences])
        ])
        next_states = tuple([
            torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
            for x in zip(*[e.next_state for e in experiences])
            ])
        return (states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)

@dataclass
class AgentSolver:
    embedding_dim: int = 32
    action_dim: int = 10
    n_clients: int = 100
    gamma: float = 0.97
    lr: float = 1e-3
    buffer_size: int = 640
    batch_size: int = 32
    tau: float = 7.5e-3
    device: str = None
    flatten: Callable = None
    to_state_dict: Callable = None
    qnet_modelfunc: Type[nn.Module] = None
    statetrans_func: Callable = Type[BaseClusterReducer]
    param_size: int = 0
    cluster_size: int = 0
    selection_size: int = 5
    feature_dim: int = 0
    attention_dim: int = 64
    recall: int = 5
    b: int = 5
    padding: bool = False
    use_temporal_attn: bool = True
    use_spatial_attn: bool = True
    encoder: str = 'attention'
    # combo-attention (WT2/WT4/WT5); defaults reproduce master behavior
    enc_input: str = 'abs'
    progress_relative_reward: bool = False
    reward_smooth: float = 0.05
    balance_select: str = 'mean'
    qnet_name: str = 'multistep'

    @classmethod
    def from_args(cls, args):
        modelfunc = set_model_fn(args.dataset)
        param_size, flatten, unflatten = parse_modelfunc(modelfunc)
        selection_size = args.selection_size
        device = parse_devices(args.gpu_ids)
        device = device[hash(str(args.path)) % len(device)] if device else 'cpu'
        recall = getattr(args, 'recall', 5)

        from data_wrapper.base import get_dataset_config
        config = get_dataset_config(args.dataset)
        n_class = config.n_class
        if n_class is None and config.create_fn:
            config.create_fn()
            n_class = config.n_class
        if n_class is None:
            n_class = config.local_cls.n_class

        return cls(
            action_dim=n_class,
            selection_size=selection_size,
            param_size=param_size,
            cluster_size=args.cluster_size,
            n_clients=args.n_clients,
            qnet_modelfunc=set_qnet_fn(args.qnet),
            statetrans_func=set_feature_reducer(args.feature),
            feature_dim=args.feature_dim,
            flatten=flatten,
            to_state_dict=unflatten,
            device=device,
            recall=recall,
            padding=args.padding,
            use_temporal_attn=not getattr(args, 'no_temporal_attn', False),
            use_spatial_attn=not getattr(args, 'no_spatial_attn', False),
            encoder='mlp' if getattr(args, 'mlp_encoder', False) else 'attention',
            enc_input=getattr(args, 'enc_input', 'abs'),
            progress_relative_reward=getattr(args, 'progress_relative_reward', False),
            reward_smooth=getattr(args, 'reward_smooth', 0.05),
            balance_select=getattr(args, 'balance_select', 'mean'),
            qnet_name=getattr(args, 'qnet', 'multistep'),
        )

class Agent:
    def __init__(self, solver: AgentSolver):
        self.solver = solver
        self.memory = MultiStepReplayBuffer(
            capacity=self.solver.buffer_size,
            n_step=self.solver.recall,
            gamma=self.solver.gamma,
            padding=self.solver.padding
        )

        self.qnetwork_local = self.solver.qnet_modelfunc(
            d_raw_feature = self.solver.feature_dim,
            d_embedding = self.solver.embedding_dim,
            d_attention = self.solver.attention_dim,
            nclasses = self.solver.action_dim,
            use_temporal_attn=self.solver.use_temporal_attn,
            use_spatial_attn=self.solver.use_spatial_attn,
            encoder=self.solver.encoder,
            enc_input=self.solver.enc_input,
            ).to(self.solver.device)

        self.qnetwork_target = self.solver.qnet_modelfunc(
            d_raw_feature = self.solver.feature_dim,
            d_embedding = self.solver.embedding_dim,
            d_attention = self.solver.attention_dim,
            nclasses = self.solver.action_dim,
            use_temporal_attn=self.solver.use_temporal_attn,
            use_spatial_attn=self.solver.use_spatial_attn,
            encoder=self.solver.encoder,
            enc_input=self.solver.enc_input,
            ).to(self.solver.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.solver.lr, weight_decay=1e-4)

        self.epsilon_max = 1.0
        self.epsilon_min = 0.1

        self.statetransformer = self.solver.statetrans_func(self.solver.feature_dim, device=self.solver.device)
        self.statetransformer.fit(X=torch.randn((self.solver.cluster_size, self.solver.param_size)))
        self.Q_max = 1/ (1 - self.solver.gamma)

        self.logger = logging.getLogger('Agent')
        self.logger.info(f"Agent initialized. feature_dim={self.solver.feature_dim}, action_dim={self.solver.action_dim}")

        self.ce = 0.15

    def remember(self, state, action, reward):
        self.memory.add(state, action, reward)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        """Per-column z-score over the cluster axis -> 0-mean, unit-variance, O(1) scale.

        Applied to grad/scalar encoder inputs so they survive the Q-net Linear+LayerNorm;
        abs is left raw to reproduce the documented input collapse as the control."""
        x = x.float()
        if x.dim() == 1:
            std = x.std(unbiased=False)
            return (x - x.mean()) if std <= 1e-6 else (x - x.mean()) / (std + 1e-6)
        std = x.std(dim=0, unbiased=False)
        mean = x.mean(dim=0)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        return (x - mean) / std

    @torch.no_grad()
    def _build_encoder_input(self, flat_Wi: torch.Tensor, flat_glob: torch.Tensor, model_param_list):
        """Map (W_i, W_glob) -> (x_u, x_s) feature tokens per the enc_input mode.

        - abs:    x_u = statetrans(W_i);                x_s = statetrans(W_glob)   [master control]
        - grad:   x_u = zscore(statetrans(W_i - W_glob));  x_s = statetrans(W_glob)
        - scalar: x_u = zscore([||grad_i||, train_loss_i, |D_i|]) slotted into feature_dim; x_s = 0
        """
        mode = getattr(self.solver, 'enc_input', 'abs')
        fd = self.solver.feature_dim
        if mode == 'grad':
            flat_params = self.statetransformer.transform(X=flat_Wi - flat_glob.unsqueeze(0))
            flat_params = self._zscore(flat_params)
            flat_glob_param = self.statetransformer.transform(X=flat_glob.unsqueeze(0)).squeeze(0)
        elif mode == 'scalar':
            n = flat_Wi.size(0)
            grad_norm = (flat_Wi - flat_glob.unsqueeze(0)).norm(dim=-1)
            local_loss = torch.tensor([float(p.get('train_loss', p.get('local_loss', 0.0))) for p in model_param_list])
            num_samples = torch.tensor([float(p.get('num_samples', 0.0)) for p in model_param_list])
            feats = torch.zeros(n, fd)
            for col, vec in enumerate((grad_norm, local_loss, num_samples)):
                feats[:, col] = self._zscore(vec)
            flat_params = feats
            flat_glob_param = torch.zeros(fd)
        else:   # abs (master)
            flat_params = self.statetransformer.transform(X=flat_Wi)
            flat_glob_param = self.statetransformer.transform(X=flat_glob.unsqueeze(0)).squeeze(0)
        return flat_params, flat_glob_param

    @torch.no_grad()
    def dicts2state(self, step, model_param_list, current_model_param, output_dist, history) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        client_ids = torch.tensor([params['id'] for params in model_param_list],dtype=torch.long)
        flat_Wi = torch.stack([self.solver.flatten(params['model_params']) for params in model_param_list], dim=0)
        flat_glob = self.solver.flatten(current_model_param)
        flat_params, flat_glob_param = self._build_encoder_input(flat_Wi, flat_glob, model_param_list)
        if len(history) == 0:
            history_tensor = torch.zeros(self.solver.recall, self.solver.feature_dim + self.solver.selection_size)
        else:
            history_tensor = torch.stack(
                [
                    torch.cat(h, dim=0) for h in history
                ], dim=0
            )
            if history_tensor.shape[0] < self.solver.recall:
                hitory_tensor_full = torch.zeros(self.solver.recall, history_tensor.shape[1])
                hitory_tensor_full[-history_tensor.shape[0]:, :] = history_tensor
                history_tensor = hitory_tensor_full
        return client_ids, flat_params, flat_glob_param, output_dist, history_tensor

    def act(self, state, epoch):
        client_ids, features, current, dist, history = state
        client_ids = client_ids.to(dtype=torch.long).unsqueeze(0).to(self.solver.device)
        features = features.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        current = current.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        history = history.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        dist = dist.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        epsilon = linear_decay(self.epsilon_max, self.epsilon_min, 5e-3, epoch)

        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.qnetwork_local(features, current, client_ids, dist, history, mask=None)[0].squeeze(-1)
            action = torch.topk(q_values, k=self.solver.selection_size, dim=1)[1].squeeze(0).cpu()
            self.logger.debug(f"Trust action")
            behavior = 'trust'
        else:
            with torch.no_grad():
                q_values = self.qnetwork_local(features, current, client_ids, dist, history, mask=None)[0].squeeze(-1)
            action = torch.topk(q_values, k=self.solver.selection_size - 2, dim=1)[1].squeeze(0).tolist()
            action_rest = list(set(range(features.size(1))) - set(action))
            action = action + random.sample(action_rest, 2)
            action = torch.tensor(action, dtype=torch.long)
            self.logger.debug(f"Random action")
            behavior = 'explore'
        return action, epsilon, behavior

    def _learn_one_batch(self):
        states, actions, rewards, next_states = self.memory.sample(batch_size=self.solver.batch_size)

        client_ids = states[0].to(self.solver.device)
        user_emb, current_emb = states[1].to(self.solver.device), states[2].to(self.solver.device)
        dist, history = states[3].to(self.solver.device), states[4].to(self.solver.device)
        actions = actions.to(self.solver.device)
        rewards = rewards.to(self.solver.device)
        next_client_ids = next_states[0].to(self.solver.device)
        next_user_emb, next_current_emb = next_states[1].to(self.solver.device), next_states[2].to(self.solver.device)
        next_dist, next_history = next_states[3].to(self.solver.device), next_states[4].to(self.solver.device)
        if self.solver.padding:
            mask, next_mask = states[5].to(self.solver.device), next_states[5].to(self.solver.device)
        else:
            mask, next_mask = None, None

        with torch.no_grad():
            action_expected_next = self.qnetwork_local(next_user_emb, next_current_emb, next_client_ids, next_dist, next_history, next_mask)[0].squeeze(-1)
            action_expected_next = action_expected_next.topk(k=self.solver.selection_size, dim=1)[1]
            Q_targets_next = self.qnetwork_target(next_user_emb, next_current_emb, next_client_ids, next_dist, next_history, next_mask)[0].squeeze(-1)
            Q_targets_next = Q_targets_next.gather(1, action_expected_next)
            Q_targets_next = Q_targets_next.mean(dim=-1, keepdim=True)
            Q_targets = rewards.mean(dim=-1, keepdim=True) + ((self.solver.gamma**self.solver.recall) * Q_targets_next)

        Q_expected, dist_pred = self.qnetwork_local(user_emb, current_emb, client_ids, dist, history, mask)
        Q_expected = Q_expected.squeeze(-1)
        Q_expected = Q_expected.gather(1, actions)
        Q_expected = Q_expected.mean(dim=-1, keepdim=True)
        tdloss = nn.SmoothL1Loss()(Q_expected, Q_targets)
        celoss = nn.L1Loss()(dist_pred, dist)
        loss = tdloss.add(celoss, alpha=self.ce)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        self._soft_update()

        return tdloss.item(), celoss.item(), Q_expected.mean().item()

    def learn(self):
        if len(self.memory) < 1.2 * self.solver.batch_size:
            self.logger.debug(f"Not enough samples in memory to learn. Current size: {len(self.memory)}")
            return  0.0, 0.0, 0.0
        total_tdloss = 0.0
        total_celoss = 0.0
        total_q = 0.0
        for _ in range(self.solver.b):
            tdloss, celoss, q_value = self._learn_one_batch()
            total_tdloss += tdloss
            total_celoss += celoss
            total_q += q_value
        avg_tdloss = total_tdloss / self.solver.b
        avg_celoss = total_celoss / self.solver.b
        avg_q = total_q / self.solver.b
        return avg_tdloss, avg_celoss, avg_q

    def get_alpha(self, new_weight: Dict, old_weight: torch.Tensor):
        new_weight = self.solver.flatten(new_weight)
        new_weight = self.statetransformer.transform(X=new_weight.unsqueeze(0))
        weights = torch.cat((old_weight.unsqueeze(0), new_weight), dim=0).to(self.solver.device)
        theta = self.qnetwork_target.model_param_embedding(weights)
        alpha = torch.norm(theta[0] - theta[1], p=2).mul(-1).sigmoid().add(0.5).item()
        return alpha

    def _soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.solver.tau * local_param.data + (1.0 - self.solver.tau) * target_param.data)


class AgentVec(Agent):
    """WT5 Path B: per-class (vectorized) Q-learning + balance-aware selection.

    The Q-net outputs q of shape (B, N, C) (one Q̂ per client per class). TD learning
    is per-class (no mean over classes), and client selection uses a balance-aware
    score (mean over classes by default) instead of top-K on a scalar average.
    """

    def _balance_score(self, q: torch.Tensor) -> torch.Tensor:
        """(B, N, C) per-class Q̂ -> (B, N) selection score per the configured criterion."""
        if getattr(self.solver, 'balance_select', 'mean') == 'maxmin':
            # maximin: favor clients strong on their WORST class (fairness)
            return q.min(dim=-1).values
        # 'mean' control: identical objective shape as the scalar Q-net
        return q.mean(dim=-1)

    def act(self, state, epoch):
        client_ids, features, current, dist, history = state
        client_ids = client_ids.to(dtype=torch.long).unsqueeze(0).to(self.solver.device)
        features = features.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        current = current.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        history = history.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        dist = dist.to(dtype=torch.float32).unsqueeze(0).to(self.solver.device)
        epsilon = linear_decay(self.epsilon_max, self.epsilon_min, 5e-3, epoch)
        N = features.size(1)
        K = self.solver.selection_size

        with torch.no_grad():
            q = self.qnetwork_local(features, current, client_ids, dist, history, mask=None)[0].squeeze(-1)  # (1,N,C)
        score = self._balance_score(q)  # (1,N)
        if random.random() > epsilon:
            action = score.topk(k=K, dim=1)[1].squeeze(0).cpu()
            self.logger.debug(f"Trust action (vec)")
            behavior = 'trust'
        else:
            action = score.topk(k=K - 2, dim=1)[1].squeeze(0).tolist()
            action_rest = list(set(range(N)) - set(action))
            action = torch.tensor(action + random.sample(action_rest, 2), dtype=torch.long)
            self.logger.debug(f"Random action (vec)")
            behavior = 'explore'
        return action, epsilon, behavior

    def _learn_one_batch(self):
        states, actions, rewards, next_states = self.memory.sample(batch_size=self.solver.batch_size)

        client_ids = states[0].to(self.solver.device)
        user_emb, current_emb = states[1].to(self.solver.device), states[2].to(self.solver.device)
        dist, history = states[3].to(self.solver.device), states[4].to(self.solver.device)
        actions = actions.to(self.solver.device)
        rewards = rewards.to(self.solver.device)  # (B, C) per-class
        assert rewards.dim() == 2 and rewards.size(-1) == self.solver.action_dim, \
            f"AgentVec expects per-class reward (B,C); got {tuple(rewards.shape)}"
        next_client_ids = next_states[0].to(self.solver.device)
        next_user_emb, next_current_emb = next_states[1].to(self.solver.device), next_states[2].to(self.solver.device)
        next_dist, next_history = next_states[3].to(self.solver.device), next_states[4].to(self.solver.device)
        if self.solver.padding:
            mask, next_mask = states[5].to(self.solver.device), next_states[5].to(self.solver.device)
        else:
            mask, next_mask = None, None

        K = self.solver.selection_size
        C = self.solver.action_dim

        with torch.no_grad():
            # double Q: pick next action by balance score on LOCAL net, evaluate on TARGET net
            q_next_local = self.qnetwork_local(next_user_emb, next_current_emb, next_client_ids, next_dist, next_history, next_mask)[0].squeeze(-1)  # (B,N,C)
            next_score = self._balance_score(q_next_local)  # (B,N)
            next_actions = next_score.topk(k=K, dim=1)[1]  # (B,K)
            q_next_target = self.qnetwork_target(next_user_emb, next_current_emb, next_client_ids, next_dist, next_history, next_mask)[0].squeeze(-1)  # (B,N,C)
            q_next_target = q_next_target.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, C))  # (B,K,C)
            Q_targets_next = q_next_target.mean(dim=1)  # (B,C)
            Q_targets = rewards + (self.solver.gamma ** self.solver.recall) * Q_targets_next  # (B,C)

        Q_expected, dist_pred = self.qnetwork_local(user_emb, current_emb, client_ids, dist, history, mask)
        Q_expected = Q_expected.squeeze(-1)  # (B,N,C)
        Q_expected = Q_expected.gather(1, actions.unsqueeze(-1).expand(-1, -1, C))  # (B,K,C)
        Q_expected = Q_expected.mean(dim=1)  # (B,C)
        tdloss = nn.SmoothL1Loss()(Q_expected, Q_targets)
        celoss = nn.L1Loss()(dist_pred, dist)
        loss = tdloss.add(celoss, alpha=self.ce)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        self._soft_update()

        return tdloss.item(), celoss.item(), Q_expected.mean().item()
