
import os
import uuid
import time
import argparse
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, Walker2dEnv


def experiment(variant):
    if variant['env'] == 'halfcheetah':
        Env = HalfCheetahEnv
    elif variant['env'] == 'hopper':
        Env = HopperEnv
    elif variant['env'] == 'walker2d':
        Env = Walker2dEnv
    else:
        raise ValueError(f"Invalid {variant['env']}")
    expl_env = NormalizedBoxEnv(Env())
    eval_env = NormalizedBoxEnv(Env())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()
    save_replay_buffer(variant, replay_buffer)


def save_replay_buffer(variant, replay_buffer):
    full_data_dict = {
        'replay_buffer/observations': replay_buffer._observations,
        'replay_buffer/actions': replay_buffer._actions,
        'replay_buffer/rewards': replay_buffer._rewards,
        'replay_buffer/terminals': replay_buffer._terminals,
        'replay_buffer/next_observations': replay_buffer._next_obs,
        'replay_buffer/env_infos': replay_buffer._env_infos,
    }
    torch.save(full_data_dict, os.path.join(variant['log_dir'], 'full_replay_buffer.pkl'))
    buffer_size = replay_buffer._size
    current_data_dict = {
        'replay_buffer/observations': replay_buffer._observations[:buffer_size],
        'replay_buffer/actions': replay_buffer._actions[:buffer_size],
        'replay_buffer/rewards': replay_buffer._rewards[:buffer_size],
        'replay_buffer/terminals': replay_buffer._terminals[:buffer_size],
        'replay_buffer/next_observations': replay_buffer._next_obs[:buffer_size],
        'replay_buffer/env_infos': replay_buffer._env_infos,
    }
    torch.save(current_data_dict, os.path.join(variant['log_dir'], 'current_replay_buffer.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker2d', choices=['halfcheetah', 'hopper', 'walker2d'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--group', default="2024052101", type=str)
    parser.add_argument('--save_path', type=str, default='~/results/corruption')
    args = parser.parse_args()
    time_tag = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f"SAC_{args.env}_{args.seed}_{time_tag}_{str(uuid.uuid4())}"
    log_dir = os.path.join(args.save_path, args.group, args.env, exp_prefix)
    log_dir = os.path.expanduser(log_dir)
    # noinspection PyTypeChecker
    variant = dict(
        log_dir=log_dir,
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    variant.update(vars(args))
    set_seed(args.seed)
    setup_logger(exp_prefix, variant, log_dir=log_dir)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
