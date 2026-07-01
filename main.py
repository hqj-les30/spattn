from servers import set_server, ServerSolver
import argparse
from pathlib import Path
from utils import set_seed

def run_classification_exp(args):
    if args.method == 'STCS' and args.recall == 0:
        args.qnet = 'singleT'
        args.recall = 1
    if args.dataset == 'har':
        args.n_clients = 21
    if args.cluster_sampler == 'onoff':
        args.padding = True
    else:
        args.padding = False
    server_solver = ServerSolver.from_args(args)
    server = set_server(args.method)(
        server_solver=server_solver
    )
    server.run()

def main(args):
    title = f"{args.method}_{args.dataset}_{args.setting}_n[{args.n_clients}]_c[{args.cluster_size}]_s[{args.selection_size}]_t{args.total_epochs}_k[{args.local_epochs}]"
    # if args.warm_up > 0:
    #     title += f"_w{args.warm_up}"
    if args.info:
        title = title + '_' + args.info

    args.path = Path(args.path) / title
    args.path.mkdir(parents=True, exist_ok=True)
    args.fraction = [float(x) for x in args.fraction.strip().split(',')]
    set_seed(args.seed)
    run_classification_exp(args)
    print(f"{args.method}{args.legend}: saved results to {args.path.absolute() / 'details.jsonl'}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="STCS experiment")
    
    # Add arguments here:

    # experiment settings
    parser.add_argument('-a', '--method', type=str, default='STCS', help="method name (default 'STCS')")
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='dataset to train')
    parser.add_argument('-n', '--n-clients', type=int, default=100, help='client number')
    parser.add_argument('-g', '--gpu-ids', type=str, default='0,1,2,3', help='devices, e.g. "0,1,2"')
    parser.add_argument('-i', '--info', type=str, default=None, help='extra information')
    parser.add_argument('-p', '--path', type=str, default='./result/', help='root path to save results')
    parser.add_argument('--seed', type=int, default=40, help='random seed')
    parser.add_argument('--workers-per-gpu', type=int, default=2, help='number of worker processes per GPU')

    # dataset settings
    parser.add_argument('--r0', type=float, default=0.95, help='the server will aggregate the cluster_size // l smallest lambda clients')
    parser.add_argument('--r1', type=float, default=1.00, help='the server will aggregate the cluster_size // l smallest lambda clients')
    parser.add_argument('-f', '--fraction', type=str, default='0.02,0.0,0.98', help='fraction of clients in each data partition')
    parser.add_argument('--setting', type=str, default='niid', help='data heterogeneity setting')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet alpha (smaller = more heterogeneous)')
    parser.add_argument('--num-shards', type=int, default=2, help='number of shards per user for label skew (fewer = more heterogeneous)')
    # federated training
    parser.add_argument('-t', '--total-epochs', type=int, default=50, help='number of the communication epochs')
    # parser.add_argument('-w', '--warm-up', type=int, default=20, help='number of the warmup epochs(no need for FedGMI)')
    parser.add_argument('-s', '--selection-size', type=int, default=5, help='number of the selected clients in each cluster')
    parser.add_argument('-k', '--local-epochs', type=int, default=3, help='number of local epochs')
    parser.add_argument('-c', '--cluster-size', type=int, default=10, help='cluster size')
    parser.add_argument('-z', '--cluster-sampler', type=str, default='clsrand', help='cluster size')

    # agent setting
    parser.add_argument('-F', '--feature-dim', type=int, default=64, help='number of local epochs')
    parser.add_argument('-q', '--qnet', type=str, default='multistep_vec', help='model structure of agent (multistep_vec=per-class Q [default]; multistep=scalar Q; noemb; singleT)')
    parser.add_argument('--feature', type=str, default='RP', help='feature extraction method')
    parser.add_argument('--recall', type=int, default=1, help='hidden dimension of agent network')
    parser.add_argument('--insert', type=int, default=0, help='hidden dimension of agent network')
    parser.add_argument('--metric', type=str, default='f1', help='feature extraction method')
    parser.add_argument('--beta', type=float, default=0.001, help='regularization coefficient (unused by STCS; kept for interface compatibility)')

    # component-level ablation flags (C2)
    parser.add_argument('--no-temporal-agg', action='store_true', help='A1c: disable temporal parameter aggregation (standard FedAvg)')
    parser.add_argument('--no-temporal-attn', action='store_true', help='A2c: disable temporal self-attention in Q-net')
    parser.add_argument('--no-spatial-attn', action='store_true', help='A3c: disable spatial (cluster) self-attention in Q-net')
    parser.add_argument('--mlp-encoder', action='store_true', help='A5c: replace attention encoder with a feed-forward MLP')

    # combo-attention method (WT2/WT4/WT5): scalar input + per-class Q are ON by default;
    # progress-relative reward is OFF by default (available via flag).
    parser.add_argument('--enc-input', type=str, default='scalar', choices=['abs', 'grad', 'scalar'],
                        help='Q-net encoder input space (scalar=[grad-norm,train_loss,|D|] [default]; abs=RP(W_i); grad=RP(W_i-W_glob))')
    parser.add_argument('--progress-relative-reward', action='store_true',
                        help='reward = score - EMA(score), removing the monotonic epoch trend (default off = EMA reward)')
    parser.add_argument('--reward-smooth', type=float, default=0.05, help='EMA factor for the progress baseline (smaller=slower)')
    parser.add_argument('--balance-select', type=str, default='mean', choices=['mean', 'maxmin'],
                        help='per-class Q scalarization for selection (mean=average over classes [default]; maxmin=favor worst-class)')
    parser.add_argument('--legend', type=str, default='', help='Legend for comparison methods')
    args = parser.parse_args()

    main(args)
