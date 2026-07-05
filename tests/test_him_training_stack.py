from collections import OrderedDict

import torch

from instinct_rl.env import VecEnv
from instinct_rl.runners.on_policy_runner import OnPolicyRunner
from instinct_rl.storage import HIMRolloutStorage


class MockHIMVecEnv(VecEnv):
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.num_envs = 3
        self.num_actions = 2
        self.num_rewards = 1
        self.max_episode_length = 32
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cfg = {}
        self.step_count = 0
        self.num_one_step_obs = 4
        self.history_steps = 2
        self.policy_dim = self.num_one_step_obs * self.history_steps
        self.critic_dim = self.num_one_step_obs + 3
        self.policy_obs = torch.zeros(self.num_envs, self.policy_dim, device=self.device)
        self.critic_obs = torch.zeros(self.num_envs, self.critic_dim, device=self.device)

    def get_obs_format(self):
        return {
            "policy": OrderedDict([("history", (self.policy_dim,))]),
            "critic": OrderedDict([("privileged", (self.critic_dim,))]),
        }

    def get_observations(self):
        return self.policy_obs.clone(), {
            "observations": {"policy": self.policy_obs.clone(), "critic": self.critic_obs.clone()}
        }

    def reset(self):
        self.policy_obs = self._make_policy_obs(0)
        self.critic_obs = self._make_critic_obs(0)
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(self, actions):
        self.step_count += 1
        self.episode_length_buf += 1
        terminal_critic = self._make_critic_obs(900 + self.step_count)
        next_policy = self._make_policy_obs(self.step_count)
        next_critic = self._make_critic_obs(self.step_count)
        dones = torch.tensor([False, self.step_count == 1, False], device=self.device)
        reset_critic = self._make_critic_obs(-100 - self.step_count)
        next_critic[dones] = reset_critic[dones]
        rewards = actions.sum(dim=-1, keepdim=True) + 0.25

        self.policy_obs = next_policy
        self.critic_obs = next_critic
        infos = {
            "observations": {"policy": next_policy.clone(), "critic": next_critic.clone()},
            "terminal_observations": {"critic": terminal_critic.clone()},
            "step": {},
        }
        return next_policy.clone(), rewards, dones, infos

    def _make_policy_obs(self, offset):
        base = torch.arange(self.num_envs * self.policy_dim, device=self.device, dtype=torch.float32)
        return (base.reshape(self.num_envs, self.policy_dim) + float(offset)) / 10.0

    def _make_critic_obs(self, offset):
        base = torch.arange(self.num_envs * self.critic_dim, device=self.device, dtype=torch.float32)
        return (base.reshape(self.num_envs, self.critic_dim) + float(offset)) / 10.0


def _him_train_cfg():
    return {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "policy": {
            "class_name": "HIMActorCritic",
            "num_one_step_obs": 4,
            "actor_hidden_dims": [16],
            "critic_hidden_dims": [16],
            "init_noise_std": 0.5,
            "estimator": {
                "enc_hidden_dims": [16, 4],
                "tar_hidden_dims": [16],
                "num_prototype": 8,
                "learning_rate": 1e-3,
            },
        },
        "algorithm": {
            "class_name": "HIMPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1e-3,
            "gamma": 0.95,
            "lam": 0.9,
            "desired_kl": None,
            "entropy_coef": 0.0,
        },
    }


def _clone_params(module):
    return [param.detach().clone() for param in module.parameters()]


def _any_param_changed(before, module):
    return any(not torch.allclose(old, new.detach()) for old, new in zip(before, module.parameters()))


def test_him_runner_rollout_update_and_checkpoint(tmp_path):
    torch.manual_seed(7)
    env = MockHIMVecEnv()
    runner = OnPolicyRunner(env, _him_train_cfg(), log_dir=None, device="cpu")

    assert isinstance(runner.alg.storage, HIMRolloutStorage)
    assert runner.alg.storage.observations.device.type == "cpu"
    assert runner.alg.storage.critic_observations.shape == (2, env.num_envs, env.critic_dim)
    assert runner.alg.storage.next_privileged_observations.shape == (2, env.num_envs, env.critic_dim)

    obs, extras = env.get_observations()
    critic_obs = extras["observations"]["critic"]
    for _ in range(2):
        obs, critic_obs, _, _, _ = runner.rollout_step(obs, critic_obs)

    expected_terminal_critic = env._make_critic_obs(901)[1]
    assert torch.allclose(runner.alg.storage.next_privileged_observations[0, 1], expected_terminal_critic)

    runner.alg.compute_returns(critic_obs)
    actor_params = _clone_params(runner.alg.actor_critic.actor)
    estimator_params = _clone_params(runner.alg.actor_critic.estimator)

    losses, stats = runner.alg.update(0)

    assert torch.isfinite(losses["surrogate_loss"])
    assert torch.isfinite(losses["value_loss"])
    assert torch.isfinite(stats["estimation_loss"])
    assert torch.isfinite(stats["swap_loss"])
    assert _any_param_changed(actor_params, runner.alg.actor_critic.actor)
    assert _any_param_changed(estimator_params, runner.alg.actor_critic.estimator)
    assert runner.alg.storage.step == 0

    checkpoint_path = tmp_path / "model.pt"
    runner.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "estimator_optimizer_state_dict" in checkpoint

    loaded_runner = OnPolicyRunner(MockHIMVecEnv(), _him_train_cfg(), log_dir=None, device="cpu")
    loaded_runner.load(checkpoint_path)
    for saved, loaded in zip(runner.alg.actor_critic.parameters(), loaded_runner.alg.actor_critic.parameters()):
        assert torch.allclose(saved, loaded)
