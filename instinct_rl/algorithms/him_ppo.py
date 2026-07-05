# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage import HIMRolloutStorage
from instinct_rl.utils.utils import get_subobs_size


class HIMPPO(PPO):
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        self.transition = HIMRolloutStorage.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        self.storage = HIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            device=self.device,
        )

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        if next_critic_obs is None:
            next_critic_obs = next_obs
        terminal_observations = infos.get("terminal_observations", {})
        terminal_critic_obs = terminal_observations.get("critic", None)
        if terminal_critic_obs is not None:
            next_critic_obs = next_critic_obs.clone()
            next_critic_obs[dones.to(torch.bool)] = terminal_critic_obs[dones.to(torch.bool)]
        self.transition.next_critic_observations = next_critic_obs
        return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    def compute_losses(self, minibatch):
        estimation_loss, swap_loss = self.actor_critic.estimator.update(
            minibatch.obs,
            minibatch.next_critic_obs,
            lr=self.learning_rate,
        )
        losses, inter_vars, stats = super().compute_losses(minibatch)
        stats["estimation_loss"] = estimation_loss
        stats["swap_loss"] = swap_loss
        return losses, inter_vars, stats

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["estimator_optimizer_state_dict"] = self.actor_critic.estimator.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "estimator_optimizer_state_dict" in state_dict:
            self.actor_critic.estimator.optimizer.load_state_dict(state_dict["estimator_optimizer_state_dict"])
        else:
            print("Warning: estimator optimizer state dict is not found, the state dict is not loaded.")
