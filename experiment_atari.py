import d4rl_atari
import traceback
import argparse
import ast
import os

from decision_transformer.atari_utils.core import train
from logger import init_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_type', type=str, default='QT')
    parser.add_argument('--exp_name', type=str, default='Atari')
    parser.add_argument('--seed', type=int, default=123)
    # environment parameters
    parser.add_argument('--env', type=str, default='breakout', choices=['breakout', 'pong', 'seaquest', 'qbert'])
    parser.add_argument('--dataset', type=str, default='mixed', choices=['medium', 'expert', 'mixed'])
    parser.add_argument('--stack_frame', type=int, default=4)
    parser.add_argument('--vec_envs', type=int, default=1, help='Vector environments')
    # algorithm parameters
    parser.add_argument("--k_rewards", action='store_true', default=False)
    parser.add_argument("--use_discount", action='store_true', default=False)
    parser.add_argument("--rtg_no_q", action='store_true', default=False)
    parser.add_argument("--infer_no_q", action='store_true', default=False)
    parser.add_argument("--infer_normal", action='store_true', default=False)
    parser.add_argument("--pred_s", action='store_true', default=False)
    parser.add_argument("--pred_r", action='store_true', default=False)
    parser.add_argument("--use_rtg", action='store_true', default=False)
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--quantile", default=0.0, type=float)
    # model parameters
    parser.add_argument('--drop_p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--context_len', type=int, default=30, help='Context length')
    parser.add_argument('--reward_scale', type=float, default=1, help='Reward scale')
    parser.add_argument('--max_timestep', type=str, default=4096, help='Maximum timestep')
    parser.add_argument('--cnn_channels', type=str, default='[32, 64, 64]', help='CNN channels')
    parser.add_argument('--cnn_kernels', type=str, default='[8, 4, 3]', help='CNN kernels')
    parser.add_argument('--cnn_strides', type=str, default='[4, 2, 1]', help='CNN strides')
    parser.add_argument('--cnn_paddings', type=str, default='[0, 0, 0]', help='CNN paddings')
    parser.add_argument('--mlp_embedding', type=int, default=0, help='MLP embedding')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    # traning parameters
    parser.add_argument('--train_steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluation interval')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--warmup_tokens', type=float, default=375e6, help='Number of warmup tokens')
    parser.add_argument('--final_tokens', type=float, default=260e9, help='Number of final tokens')
    # dataset attack
    parser.add_argument('--dataset_path', type=str, default='/apdcephfs/share_1563664/ztjiaweixu/datasets')
    parser.add_argument("--down_sample", default=1, type=int, choices=[0, 1])
    parser.add_argument("--use_collected", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_suffix", type=str, default="3_20000_none_collect")
    parser.add_argument('--sample_ratio', default=0.1, type=float)
    parser.add_argument('--corruption_agent', default="IQL", type=str)
    parser.add_argument('--corruption_mode', default="none", type=str, choices=["none", "random", "adversarial"])
    parser.add_argument('--corruption_seed', default=0, type=int)
    parser.add_argument('--corruption_obs', default=0.0, type=float)
    parser.add_argument('--corruption_act', default=0.0, type=float)
    parser.add_argument('--corruption_rew', default=0.0, type=float)
    parser.add_argument('--corruption_rew2', default=0.0, type=float)
    parser.add_argument('--corruption_rate', default=0.3, type=float)
    parser.add_argument('--corruption_step', default=1, type=int)
    parser.add_argument('--froce_attack', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_original', default=0, type=int, choices=[0, 1])
    parser.add_argument('--same_index', default=0, type=int, choices=[0, 1])
    # other parameters
    parser.add_argument('--group', default="2024052101", type=str)
    parser.add_argument('--save_path', type=str, default='~/results/corruption')
    args = parser.parse_args()

    os.environ['D4RL_DATASET_DIR'] = os.path.join(args.dataset_path, 'd4rl_atari')

    # Convert string representations of lists to actual lists   
    args.cnn_channels = ast.literal_eval(args.cnn_channels)
    args.cnn_kernels = ast.literal_eval(args.cnn_kernels)
    args.cnn_strides = ast.literal_eval(args.cnn_strides)
    args.cnn_paddings = ast.literal_eval(args.cnn_paddings)

    # context length
    if args.env == 'pong':
        args.context_len = 50
    else:
        args.context_len = 30

    # target return
    if args.env == 'breakout':
        args.rtg_target = [90]
    elif args.env == 'pong':
        args.rtg_target = [20]
    elif args.env == 'seaquest':
        args.rtg_target = [1450]
    elif args.env == 'qbert':
        args.rtg_target = [2500]
    
    logger = init_logger(args)
    try:
        train(args, logger=logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")