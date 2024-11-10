import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import torch
import gym
import d4rl_atari
import numpy as np
import torch.nn.functional as F
import utils

from torch.optim.lr_scheduler import _LRScheduler
from buffer import AtariBuffer
from model import DecisionTransformer
from copy import deepcopy
from gym.vector import SyncVectorEnv
from tqdm import tqdm
from get_hns import get_normalized_score
from model import DecisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLRScheduler(_LRScheduler): #follows the setup of original DT atari code
    def __init__(self, optimizer, warmup_tokens, final_tokens, last_epoch=-1):
        self.tokens = 0
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress = float(self.tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [lr * lr_mult for lr in self.base_lrs]

    def step(self, tokens=None):
        if tokens is not None:
            self.tokens += tokens
        super().step()

@torch.no_grad()
def eval_fn(env_name, env: gym.vector.VectorEnv, model: DecisionTransformer, rtg_target: float, outputs: dict):
    # parallel evaluation with vectorized environment
    model.eval()
    
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=bool)

    max_timestep = model.max_timestep
    context_len = model.context_len
    timesteps = torch.arange(max_timestep, device=device)
    state = env.reset()
    
    states = utils.TorchDeque(maxlen=context_len, device=device, dtype=torch.float32)
    actions = torch.zeros((episodes, max_timestep), dtype=torch.long, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0

    while not done_flags.all() and timestep < model.max_timestep:
        states.append(torch.from_numpy(state))
        rewards_to_go[:, timestep] = reward_to_go - torch.from_numpy(returns).to(device).unsqueeze(-1)
        obs_index = torch.arange(max(0, timestep-context_len+1), timestep+1)
        action_preds = model.forward(states.to_tensor(),
                                        actions[:, obs_index],
                                        rewards_to_go[:, obs_index], # drop rewards
                                        timesteps[None, obs_index],
                                        ) 
        action = action_preds[:, -1].argmax(dim=-1).detach()
        actions[:, timestep] = action
        state, reward, dones, _ = env.step(action.cpu().numpy())
        returns += reward * ~done_flags
        done_flags = np.bitwise_or(done_flags, dones)
        timestep += 1
        normalized_returns = get_normalized_score(env_name, returns)

    # return np.mean(returns), np.std(returns), np.mean(normalized_returns), np.std(normalized_returns)
    outputs[f"evalutation/target_{rtg_target}_return_mean"] = np.mean(returns)
    outputs[f"evalutation/target_{rtg_target}_return_std"] = np.std(returns)
    outputs[f"evalutation/target_{rtg_target}_normalized_return_mean"] = np.mean(normalized_returns)
    outputs[f"evalutation/target_{rtg_target}_normalized_return_std"] = np.std(normalized_returns)
    # return outputs

def train(cfg, logger):

    logger.info(f"instantiating environment {cfg.env} with dataset {cfg.dataset}")
    env_name = f'{cfg.env.lower()}-{cfg.dataset}-v0'
    env = gym.make(env_name, stack=cfg.stack_frame)
    data_env = gym.make(env_name, stack=False)
    eval_env = SyncVectorEnv([lambda: deepcopy(env) for _ in range(cfg.eval_episodes)])
    utils.set_seed_everywhere(eval_env, cfg.seed)
    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)

    logger.info("initializing buffer")
    buffer = AtariBuffer(
        env=data_env, dataset_type=cfg.dataset, context_len=cfg.context_len,
        stack_frame=cfg.stack_frame, sample_ratio=cfg.sample_ratio, seed=cfg.seed
    )

    logger.info("initializing model")
    model = DecisionTransformer(
        state_dim=state_dim, action_dim=action_dim, context_len=cfg.context_len, drop_p=cfg.drop_p,
        action_space=eval_env.action_space[0], reward_scale=cfg.reward_scale, max_timestep=cfg.max_timestep,
        embed_dim=cfg.embed_dim, n_layer=cfg.n_layer, n_head=cfg.n_head, mlp_embedding=cfg.mlp_embedding,
        cnn_channels=cfg.cnn_channels, cnn_kernels=cfg.cnn_kernels, cnn_strides=cfg.cnn_strides, cnn_paddings=cfg.cnn_paddings,
    ).to(device)
    trainable_parameters = 0
    total_parameters = 0
    for name, param in model.named_parameters():
        logger.info(f"{name}: requires_grad={param.requires_grad}")
        if param.requires_grad:
            trainable_parameters += param.numel()
        total_parameters += param.numel()
    logger.info(f"Total parameters: {total_parameters}, trainable parameters: {trainable_parameters}")
    logger.info(str(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CustomLRScheduler(optimizer, cfg.warmup_tokens, cfg.final_tokens)
    logger.info(f"Training seed {cfg.seed} for {cfg.train_steps} timesteps with {env_name} {buffer.dataset_type.title()} dataset")

    best_reward = -np.inf
    tokens = 0
    outputs = {}
    action_losses = []
    progress_bar = tqdm(range(1, cfg.train_steps + 1))

    eval_log = {}
    for target in cfg.rtg_target:
        eval_fn(cfg.env, eval_env, model, target, eval_log)
    logger.record("timestep", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)

    for timestep in progress_bar:
        states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        action_logits = model.forward(states, actions, rewards_to_go, timesteps)
        action_logits = action_logits[mask]
        action_loss = F.cross_entropy(action_logits, actions[mask].detach().to(dtype=torch.long))
        
        loss = action_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += (actions > 0).sum().item()
        scheduler.step(tokens)
        action_losses.append(action_loss.detach().cpu().item())
        progress_bar.set_postfix({"loss": action_losses[-1], "lr": scheduler.get_lr()[0]})

        if timestep % cfg.eval_interval == 0:
            for target in cfg.rtg_target:
                eval_fn(cfg.env, eval_env, model, target, outputs)
                eval_mean = outputs[f"evalutation/target_{target}_normalized_return_mean"]
                if eval_mean > best_reward:
                    best_reward = eval_mean
                    # model.save(f'best_train_seed_{cfg.seed}' if timestep <= cfg.train_steps else f'best_finetune_seed_{cfg.seed}')
                    logger.info(f'Seed: {cfg.seed}, Save best model at eval mean {best_reward:.4f} and step {timestep} with rtg target {target}')
            
            outputs[f"training/loss_mean"] = np.mean(action_losses)
            outputs[f"training/loss_std"] = np.std(action_losses)

            logger.record("timestep", timestep)
            for k, v in outputs.items():
                logger.record(k, v)
            logger.dump(timestep)

            outputs = {}
            action_losses = []

        # if timestep == cfg.train_steps:
        #     model.save(f'final_train_seed_{cfg.seed}')
        #     model.load(f'best_train_seed_{cfg.seed}')

    logger.info(f"Finish training seed {cfg.seed} with average eval mean: {eval_mean}")
    eval_env.close()
    return eval_mean