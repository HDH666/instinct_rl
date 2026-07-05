# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from instinct_rl.modules.actor_critic import get_activation
from instinct_rl.modules.him_estimator import HIMEstimator
from instinct_rl.utils.utils import get_subobs_size


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_format,
        num_actions,
        num_one_step_obs=None,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        estimator=dict(),
        num_rewards=1,
        **kwargs,
    ):
        if kwargs:
            print(
                "HIMActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        if num_rewards != 1:
            raise ValueError("HIMActorCritic currently supports one critic value head.")
        super().__init__()

        activation = get_activation(activation)
        self.obs_format = obs_format
        self.num_actor_obs = get_subobs_size(obs_format["policy"])
        self.num_critic_obs = get_subobs_size(obs_format.get("critic", obs_format["policy"]))
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs or self._infer_num_one_step_obs(obs_format["policy"])
        if self.num_actor_obs % self.num_one_step_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by num_one_step_obs "
                f"({self.num_one_step_obs})."
            )
        self.history_size = int(self.num_actor_obs / self.num_one_step_obs)

        estimator_cfg = estimator.copy()
        estimator_cfg.setdefault("temporal_steps", self.history_size)
        estimator_cfg.setdefault("num_one_step_obs", self.num_one_step_obs)
        self.estimator = HIMEstimator(**estimator_cfg)

        mlp_input_dim_a = self.num_one_step_obs + 3 + self.estimator.num_latent
        mlp_input_dim_c = self.num_critic_obs

        self.actor = self._build_mlp(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)
        self.critic = self._build_mlp(mlp_input_dim_c, 1, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator: {self.estimator.encoder}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _build_mlp(input_dim, output_dim, hidden_dims, activation):
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation]
        return nn.Sequential(*layers)

    @staticmethod
    def _infer_num_one_step_obs(policy_obs_format):
        if len(policy_obs_format) == 1:
            shape = next(iter(policy_obs_format.values()))
            if len(shape) == 2:
                return shape[-1]
        raise ValueError("HIMActorCritic requires num_one_step_obs when policy observation format is flat.")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:, : self.num_one_step_obs], vel, latent), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, obs_history, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        vel, latent = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:, : self.num_one_step_obs], vel, latent), dim=-1)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))
