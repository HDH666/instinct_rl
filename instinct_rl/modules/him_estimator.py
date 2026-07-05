# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic import get_activation


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=[128, 64, 16],
        tar_hidden_dims=[128, 64],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=32,
        temperature=3.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "HIMEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers += [nn.Linear(enc_input_dim, hidden_dim), activation]
            enc_input_dim = hidden_dim
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for hidden_dim in tar_hidden_dims:
            tar_layers += [nn.Linear(tar_input_dim, hidden_dim), activation]
            tar_input_dim = hidden_dim
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3].detach()
        next_obs = next_critic_obs.detach()[:, 3 : self.num_one_step_obs + 3]

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight.data.clone(), dim=-1, p=2))

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.detach(), swap_loss.detach()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    q = torch.exp(out / eps).T
    num_prototypes, batch_size = q.shape
    q /= q.sum()

    for _ in range(iters):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= num_prototypes
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch_size
    return (q * batch_size).T
