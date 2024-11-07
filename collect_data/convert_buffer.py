import argparse
import re
import os

import h5py
import torch
import numpy as np

itr_re = re.compile(r'itr_(?P<itr>[0-9]+).pkl')

def load(pklfile):
    params = torch.load(pklfile)
    env_infos = params['replay_buffer/env_infos']
    results = { 
        'observations': params['replay_buffer/observations'],
        'next_observations': params['replay_buffer/next_observations'],
        'actions': params['replay_buffer/actions'],
        'rewards': params['replay_buffer/rewards'],
        'terminals': params['replay_buffer/terminals'],
        # 'terminals': env_infos['terminal'].squeeze(),
        # 'timeouts': env_infos['timeout'].squeeze(),
        # 'infos/action_log_probs': env_infos['action_log_prob'].squeeze(),
    }
    if 'qpos' in env_infos:
        results['infos/qpos'] = env_infos['qpos']
        results['infos/qvel'] = env_infos['qvel']
    return results

def get_pkl_itr(pklfile):
    match = itr_re.search(pklfile)
    if match:
        return match.group('itr')
    raise ValueError(pklfile+" has no iteration number.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="halfcheetah")
    parser.add_argument('--pklfile', type=str, default="/apdcephfs/share_1563664/ztjiaweixu/vdt_sz/2024110701")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--full_buffer', action='store_true')
    parser.add_argument('--output_file', type=str, default='output_data')
    args = parser.parse_args()

    buffer_name = "full_replay_buffer.pkl" if args.full_buffer else "current_replay_buffer.pkl"
    pklfile = os.path.join(args.pklfile, args.env_name)
    for root, dirs, files in os.walk(pklfile):
        if f"SAC_{args.env_name}_{args.seed}_" in root:
            pklfile = os.path.join(root, buffer_name)
            break

    data = load(pklfile)

    output_file = os.path.join(args.output_file, f"{args.env_name}_{args.seed}_{args.num_data}_buffer.hdf5")
    hfile = h5py.File(output_file, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')
    hfile['metadata/algorithm'] = np.string_('SAC')
    hfile['metadata/iteration'] = np.array([get_pkl_itr(args.pklfile)], dtype=np.int32)[0]
    hfile.close()
