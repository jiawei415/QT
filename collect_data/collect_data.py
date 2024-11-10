import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import re

import h5py
import torch
import gym
import d4rl
import numpy as np

from rlkit.torch import pytorch_util as ptu
from attacker import Evaluation_Attacker

itr_re = re.compile(r'itr_(?P<itr>[0-9]+).pkl')

ENV_NAMES = {
    'HalfCheetah-v2': 'halfcheetah',
    'Hopper-v2': 'hopper',
    'Walker2d-v2': 'walker2d',
}

def load(pklfile):
    params = torch.load(pklfile)
    return params['trainer/policy']

def load_qf(pklfile):
    params = torch.load(pklfile)
    return params['trainer/qf1']

def get_pkl_itr(pklfile):
    match = itr_re.search(pklfile)
    if match:
        return match.group('itr')
    else:
        return "last"
    # raise ValueError(pklfile+" has no iteration number.")

def get_policy_wts(params):
    out_dict = {
        'fc0/weight': params.fcs[0].weight.data.cpu().numpy(),
        'fc0/bias': params.fcs[0].bias.data.cpu().numpy(),
        'fc1/weight': params.fcs[1].weight.data.cpu().numpy(),
        'fc1/bias': params.fcs[1].bias.data.cpu().numpy(),
        'last_fc/weight': params.last_fc.weight.data.cpu().numpy(),
        'last_fc/bias': params.last_fc.bias.data.cpu().numpy(),
        'last_fc_log_std/weight': params.last_fc_log_std.weight.data.cpu().numpy(),
        'last_fc_log_std/bias': params.last_fc_log_std.bias.data.cpu().numpy(),
    }
    return out_dict

def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        qpos = [],
        qvel = []
    )
    return data

def rollout(policy, env_name, max_path, num_data, random=False, attacker=None):
    env = gym.make(env_name)

    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0 
    done = False
    s = env.reset()
    if attacker is not None:
        s = attacker.attack_obs(s)
    while len(data['rewards']) < num_data:


        if random:
            a = env.action_space.sample()
            logprob = np.log(1.0 / np.prod(env.action_space.high - env.action_space.low))
        else:
            torch_s = ptu.from_numpy(np.expand_dims(s, axis=0)).to(ptu.device)
            distr = policy.forward(torch_s)
            a = distr.sample()
            logprob = distr.log_prob(a).detach().cpu().numpy()
            a = ptu.get_numpy(a).squeeze()

        #mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        try:
            ns, rew, done, infos = env.step(a)
        except:
            print('lost connection')
            env.close()
            env = gym.make(env_name)
            s = env.reset()
            traj_data = get_reset_data()
            t = 0
            _returns = 0
            continue

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True


        traj_data['observations'].append(s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(ns)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(logprob)
        traj_data['qpos'].append(qpos)
        traj_data['qvel'].append(qvel)

        s = ns
        if terminal or timeout:
            print('Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d' % (t, _returns, len(data['rewards']), num_data))
            s = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()
        
        if attacker is not None:
            s = attacker.attack_obs(s)
    
    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(np.bool),
        timeouts=np.array(data['timeouts']).astype(np.bool)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)
    new_data['infos/qpos'] = np.array(data['qpos']).astype(np.float32)
    new_data['infos/qvel'] = np.array(data['qvel']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="HalfCheetah-v2")
    parser.add_argument('--pklfile', type=str, default="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz/2024110701")
    parser.add_argument('--output_file', type=str, default='/apdcephfs/share_1563664/ztjiaweixu/datasets/collected')
    parser.add_argument('--max_path', type=int, default=1000)
    parser.add_argument('--num_data', type=int, default=20000)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--eval_attack', action='store_true')
    parser.add_argument('--eval_attack_eps', default=0.1, type=float)
    parser.add_argument('--eval_attack_mode', default='random', type=str, choices=['random', 'action_diff'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    env_name = ENV_NAMES[args.env]
    pklfile = os.path.join(args.pklfile, env_name)
    for root, dirs, files in os.walk(pklfile):
        if f"SAC_{env_name}_{args.seed}_" in root:
            pklfile = os.path.join(root, "params.pkl")
            break

    policy = None
    if not args.random:
        policy = load(pklfile)

    attacker = None
    attack_name = "none"
    if args.eval_attack:
        env = gym.make(args.env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        attacker = Evaluation_Attacker(
            policy, None, args.eval_attack_eps, state_dim, action_dim, None, args.eval_attack_mode
        )
        attack_name = f"{args.eval_attack_mode}_{args.eval_attack_eps}"

    data = rollout(policy, args.env, max_path=args.max_path, num_data=args.num_data, random=args.random, attacker=attacker)

    os.makedirs(args.output_file, exist_ok=True)
    output_file = os.path.join(args.output_file, f"{env_name}_{args.seed}_{args.num_data}_{attack_name}_collect.hdf5")
    hfile = h5py.File(output_file, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    if args.random:
        pass
    else:
        hfile['metadata/algorithm'] = np.string_('SAC')
        hfile['metadata/iteration'] = np.string_(get_pkl_itr(args.pklfile))
        hfile['metadata/policy/nonlinearity'] = np.string_('relu')
        hfile['metadata/policy/output_distribution'] = np.string_('tanh_gaussian')
        for k, v in get_policy_wts(policy).items():
            hfile['metadata/policy/'+k] = v
    hfile.close()
    print(f"Saved to {output_file}")
