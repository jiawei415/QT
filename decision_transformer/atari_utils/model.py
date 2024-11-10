import os
import math
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config

def get_cnn_output_dim(input_size, kernel_size, stride, padding):
    return math.floor((input_size - kernel_size + 2 * padding) / stride + 1)


def cnn(input_shape, input_channel, hidden_channels, kernel_sizes, strides, paddings, hidden_dim, output_activation=nn.Identity()):
    layers = []
    embedding_h, embedding_w = input_shape
    for i in range(len(hidden_channels)):
        layers.append(nn.Conv2d(input_channel, hidden_channels[i], kernel_sizes[i], strides[i], paddings[i]))
        layers.append(nn.ReLU())
        input_channel = hidden_channels[i]
        embedding_h = get_cnn_output_dim(embedding_h, kernel_sizes[i], strides[i], paddings[i])
        embedding_w = get_cnn_output_dim(embedding_w, kernel_sizes[i], strides[i], paddings[i])
    layers.append(nn.Flatten())
    layers.append(nn.Linear(embedding_h * embedding_w * hidden_channels[-1], hidden_dim))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.activation = nn.GELU()
    def forward(self, x):
        hidden = self.fc1(x)
        residual = hidden
        hidden = self.activation(hidden)
        out = self.fc2(hidden)
        out += residual
        return out


class ResidualEmbedBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.fc1 = torch.nn.Embedding(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.activation = nn.GELU()
    def forward(self, x):
        hidden = self.fc1(x)
        residual = hidden
        hidden = self.activation(hidden)
        out = self.fc2(hidden)
        out += residual
        return out


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, context_len, drop_p,
                 action_space, reward_scale, max_timestep,
                 embed_dim, n_layer, n_head, mlp_embedding,
                 cnn_channels, cnn_kernels, cnn_strides, cnn_paddings,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        # transformer blocks
        self.context_len = context_len
  
        # minGPT
        print("Loading with random weights on minGPT model")
        config = GPT2Config(
            vocab_size=1, # we don't need the vocab size
            n_layer=n_layer,
            n_head=n_head,
            n_embed=embed_dim,
            embd_pdrop=drop_p,
        )
        self.transformer = GPT2LMHeadModel(config=config)
        hidden_dim = config.n_embd
        self.hidden_dim = config.n_embd

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        # self.embed_dropstep = nn.Embedding(max_timestep, hidden_dim)
        if mlp_embedding:
            self.embed_rtg = ResidualBlock(1, hidden_dim)
            self.embed_action = ResidualEmbedBlock(action_dim, hidden_dim)
            # self.embed_state = ResidualBlock(np.prod(state_dim), hidden_dim)
            self.embed_state = nn.Sequential(
                cnn(state_dim[1:], state_dim[0], cnn_channels, cnn_kernels, cnn_strides, cnn_paddings, hidden_dim),
                ResidualBlock(hidden_dim, hidden_dim),
            )
        else:
            self.embed_rtg = torch.nn.Linear(1, hidden_dim)
            self.embed_action = torch.nn.Embedding(action_dim, hidden_dim)
            self.embed_state = cnn(state_dim[1:], state_dim[0], cnn_channels, cnn_kernels, cnn_strides, cnn_paddings, hidden_dim)

        # prediction heads
        self.predict_action = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.action_space = action_space
        self.reward_scale = reward_scale

        self.max_timestep = max_timestep
        # self.drop_aware = drop_aware

    def _norm_reward_to_go(self, reward_to_go):
        return reward_to_go / self.reward_scale

    # def __repr__(self):
    #     return "DecisionTransformer"
    
    def freeze_trunk(self):
        freezed_models = [self.embed_state, self.embed_action, self.embed_rtg, self.embed_timestep, self.blocks, self.embed_ln]
        for model in freezed_models:
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, states, actions, rewards_to_go, timesteps, return_embeddings=False):
        states = states / 255.0
        rewards_to_go = self._norm_reward_to_go(rewards_to_go)
        batch_size, context_len = states.shape[:2]

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states.reshape(-1, *self.state_dim)).reshape(batch_size, context_len, -1) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(rewards_to_go) + time_embeddings
        
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * context_len, self.hidden_dim)

        h = self.embed_ln(h)
        
        if return_embeddings:
            all_embs = h

        # transformer and prediction
        h = self.transformer(inputs_embeds=h, output_hidden_states=True)["hidden_states"][-1]

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(batch_size, context_len, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        action_logits = self.predict_action(h[:, 1])  # predict action given r, s

        if return_embeddings:
            return action_logits, all_embs
        else:
            return action_logits
    
    def save(self, save_name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.state_dict(), os.path.join('models', f'{save_name}.pt'))
    
    def load(self, load_name):
        self.load_state_dict(torch.load(os.path.join('models', f'{load_name}.pt')))