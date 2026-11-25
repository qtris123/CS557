#!/usr/bin/env python3
"""
Unit Tests & Andy Jones Probe Environments for CS 557 HW2
Verifies REINFORCE, PPO, and SAC correctness before full cluster training.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch

if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(1)

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reinforce import REINFORCEAgent
from ppo import PPOAgent, ActorNetwork, RunningRewardNorm
from sac import SACAgent, ReplayBuffer
import utils


# ---------------------------------------------------------------------------
# Andy Jones Probe Environments (adapted for continuous action space [-1, 1])
# ---------------------------------------------------------------------------
class _BaseProbeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def render(self):
        return np.zeros((16, 16, 3), dtype=np.uint8)


class ProbeEnv1(_BaseProbeEnv):
    """Zero obs, 1 step, constant +1 reward -> tests value function learning."""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 1.0, True, False, {}


class ProbeEnv3(_BaseProbeEnv):
    """2-step deterministic chain: obs=[0,0] -> obs=[1,0] with +1 reward at end -> tests discounting."""
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            return np.array([1.0, 0.0], dtype=np.float32), 0.0, False, False, {}
        return np.array([0.0, 0.0], dtype=np.float32), 1.0, True, False, {}


class ProbeEnv4(_BaseProbeEnv):
    """Zero obs, 1 step, reward = action[0] -> tests policy optimization direction."""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        r = float(np.clip(action[0], -1.0, 1.0))
        return np.zeros(2, dtype=np.float32), r, True, False, {}


class ProbeEnv5(_BaseProbeEnv):
    """Obs in {[-1,0], [+1,0]}, 1 step, reward = obs[0] * action[0] -> tests policy-obs coupling."""
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.obs = np.array([1.0, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sign = 1.0 if np.random.rand() > 0.5 else -1.0
        self.obs = np.array([sign, 0.0], dtype=np.float32)
        return self.obs.copy(), {}

    def step(self, action):
        r = float(self.obs[0] * np.clip(action[0], -1.0, 1.0))
        return self.obs.copy(), r, True, False, {}


class ProbeTruncEnv(_BaseProbeEnv):
    """2-step env: step 1 normal, step 2 truncated (not terminated) -> tests truncation handling."""
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.2, -0.3], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            return np.array([0.5, 0.1], dtype=np.float32), 1.0, False, False, {}
        return np.array([0.8, 0.4], dtype=np.float32), 2.0, False, True, {}


for pid, cls in [
    ("Probe1-v0", ProbeEnv1),
    ("Probe3-v0", ProbeEnv3),
    ("Probe4-v0", ProbeEnv4),
    ("Probe5-v0", ProbeEnv5),
    ("ProbeTrunc-v0", ProbeTruncEnv),
]:
    if pid not in gym.envs.registry:
        gym.register(id=pid, entry_point=cls)


def test_math_units():
    print("Running mathematical unit tests...")
    orig_cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try:
        # 1. REINFORCE discounted returns
        r_agent = REINFORCEAgent("MountainCarContinuous-v0", gamma=0.9, seed=0)
        rets = r_agent.calculate_discounted_returns([1.0, 2.0, 3.0])
        expected = np.array([1.0 + 0.9 * (2.0 + 0.9 * 3.0), 2.0 + 0.9 * 3.0, 3.0], dtype=np.float32)
        assert np.allclose(rets, expected, atol=1e-5), f"REINFORCE returns mismatch: {rets} vs {expected}"
        r_agent.logger.close()
        r_agent.env.close()

        # 2. PPO GAE across episode boundary (done=1 mid-rollout) and next_value bootstrap
        p_agent = PPOAgent("MountainCarContinuous-v0", gamma=0.9, gae_lambda=0.5, norm_obs=True, use_sde=True, seed=0)
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        next_val = 2.0
        adv, ret = p_agent.compute_gae(rewards, values, dones, next_val)
        # t=2: d=0 -> delta2 = 3.0 + 0.9*2.0 - 1.5 = 3.3, adv2 = 3.3, ret2 = 3.0 + 0.9*2.0 = 4.8
        # t=1: d=1 -> delta1 = 2.0 + 0 - 1.0 = 1.0, adv1 = 1.0, ret1 = 2.0
        # t=0: d=0 -> delta0 = 1.0 + 0.9*1.0 - 0.5 = 1.4, adv0 = 1.4 + 0.9*0.5*1.0 = 1.85, ret0 = 1.0 + 0.9*2.0 = 2.8
        assert np.allclose(adv, [1.85, 1.0, 3.3], atol=1e-5), f"GAE adv mismatch: {adv}"
        assert np.allclose(ret, [2.8, 2.0, 4.8], atol=1e-5), f"GAE ret mismatch: {ret}"

        # Check eval does NOT mutate norm_obs buffers
        p_agent.actor.update_obs_norm(np.array([0.5, -0.02], dtype=np.float32))
        before_mean = p_agent.actor.obs_norm.mean.clone()
        before_count = p_agent.actor.obs_norm.count.clone()
        _ = utils.evaluate_policy(p_agent.actor, p_agent.env.observation_space, "MountainCarContinuous-v0", episodes=1)
        assert torch.equal(before_mean, p_agent.actor.obs_norm.mean), "eval mutated obs_norm.mean!"
        assert torch.equal(before_count, p_agent.actor.obs_norm.count), "eval mutated obs_norm.count!"
        p_agent.logger.close()
        p_agent.env.close()

        # 3. ReplayBuffer wraparound & SAC alpha detach check
        buf = ReplayBuffer(4)
        for i in range(6):
            buf.push([i, i + 1], [i * 0.1], float(i), [i + 1, i + 2], float(i % 2))
        assert len(buf) == 4
        s, a, r, ns, d = buf.sample(3)
        assert s.shape == (3, 2) and a.shape == (3, 1) and r.shape == (3,)

        s_agent = SACAgent("MountainCarContinuous-v0", batch_size=4, start_steps=0, seed=0)
        for i in range(8):
            s_agent.memory.push([0.1 * i, -0.01], [0.5], 1.0, [0.1 * (i + 1), 0.0], 0.0)
        assert s_agent.alpha.requires_grad is False, "SAC self.alpha should be detached!"
        assert s_agent.log_alpha.requires_grad is True
        s_agent.update_parameters()
        assert s_agent.alpha.requires_grad is False, "SAC self.alpha should remain detached after update!"
        s_agent.logger.close()
        s_agent.env.close()

        # 4. PPO td_lambda_returns, RunningRewardNorm, 1-step explained_var, & checkpoint parity
        p_td = PPOAgent("MountainCarContinuous-v0", gamma=0.9, gae_lambda=0.5, td_lambda_returns=True, norm_obs=True, use_sde=True, seed=0)
        adv_td, ret_td = p_td.compute_gae(rewards, values, dones, next_val)
        assert np.allclose(ret_td, adv_td + values, atol=1e-5), "td_lambda_returns mismatch!"
        al1, vl1, el1 = p_td.update_policy(
            states=np.zeros((1, 2), dtype=np.float32),
            actions=np.zeros((1, 1), dtype=np.float32),
            old_log_probs=np.zeros(1, dtype=np.float32),
            advantages=np.array([1.0], dtype=np.float32),
            returns=np.array([1.0], dtype=np.float32),
            old_values=np.array([0.5], dtype=np.float32),
        )
        assert np.isfinite(p_td._last_diagnostics["explained_var"]) and np.isfinite(al1) and np.isfinite(vl1)

        p_td.actor.update_obs_norm(np.array([[0.3, -0.02], [-0.4, 0.01]], dtype=np.float32))
        ckpt_path = p_td._save_checkpoint()
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reloaded = ActorNetwork(
            2, 1,
            log_std_init=loaded["log_std_init"],
            norm_obs=loaded["norm_obs"],
            use_sde=loaded["use_sde"],
            use_tanh=loaded["use_tanh"],
        )
        reloaded.load_state_dict(loaded["actor"])
        p_td.actor.eval()
        reloaded.eval()
        test_obs = torch.tensor([[0.15, -0.01]], dtype=torch.float32)
        with torch.no_grad():
            assert torch.allclose(p_td.actor.deterministic_action(test_obs), reloaded.deterministic_action(test_obs), atol=1e-6)
        p_td.logger.close()
        p_td.env.close()

        rn = RunningRewardNorm(gamma=0.9)
        r1 = rn.normalize(2.0, done=False)
        r2 = rn.normalize(4.0, done=True)
        assert np.isfinite(r1) and np.isfinite(r2) and rn.ret == 0.0, "RunningRewardNorm failed reset!"

        # 5. SAC truncation vs true termination in ReplayBuffer during train_sac
        s_trunc = SACAgent("ProbeTrunc-v0", max_steps=4, batch_size=2, start_steps=100, save_interval=10000, seed=0)
        s_trunc.train_sac()
        assert len(s_trunc.memory) == 4
        assert np.all(s_trunc.memory.dones[:4] == 0.0), f"SAC stored truncated as terminal: {s_trunc.memory.dones[:4]}"

        s_term = SACAgent("Probe1-v0", max_steps=2, batch_size=2, start_steps=100, save_interval=10000, seed=0)
        s_term.train_sac()
        assert len(s_term.memory) == 2
        assert np.all(s_term.memory.dones[:2] == 1.0), f"SAC missed true termination: {s_term.memory.dones[:2]}"

        # 6. PPO truncation bootstrapping check: rollout captures V(next_state) on truncated step
        p_trunc = PPOAgent("ProbeTrunc-v0", gamma=0.9, max_steps=2, rollout_steps=10, save_interval=10000, seed=0)
        with torch.no_grad():
            for param in p_trunc.critic.parameters():
                param.zero_()
            p_trunc.critic.network[-1].bias.fill_(5.0)
        seen_rewards = []
        orig_gae = p_trunc.compute_gae
        def spy_gae(rew, val, dn, nv):
            seen_rewards.append(rew.copy())
            return orig_gae(rew, val, dn, nv)
        p_trunc.compute_gae = spy_gae
        p_trunc.rollout_steps = 2
        p_trunc.batch_size = 2
        p_trunc.train_ppo()
        assert len(seen_rewards) == 1 and np.allclose(seen_rewards[0], [1.0, 2.0 + 0.9 * 5.0], atol=1e-5), f"PPO truncation bootstrap failed: {seen_rewards}"

        print("  [PASS] All mathematical & invariant unit tests passed.")
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_probe_environments():
    print("Running Andy Jones probe environment tests...")
    orig_cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    try:
        # PPO on Probe1 (value network isolation): V([0,0]) -> 1.0
        ppo1 = PPOAgent("Probe1-v0", lr=1e-2, max_steps=500, gamma=0.9, rollout_steps=32, batch_size=16, update_epochs=10, save_interval=10000, seed=0)
        ppo1.train_ppo()
        v_probe1 = ppo1.estimate_value(np.array([0.0, 0.0], dtype=np.float32))
        print(f"  PPO Probe1: V([0,0])={v_probe1:.3f} (target 1.000)")
        assert abs(v_probe1 - 1.0) < 0.15, f"PPO Probe1 failed: v={v_probe1}"

        # PPO on Probe3 (discounting check): V([0,0]) -> gamma=0.9, V([1,0]) -> 1.0
        ppo3 = PPOAgent("Probe3-v0", lr=3e-3, max_steps=600, gamma=0.9, rollout_steps=64, batch_size=32, update_epochs=10, save_interval=10000, seed=0)
        ppo3.train_ppo()
        v0 = ppo3.estimate_value(np.array([0.0, 0.0], dtype=np.float32))
        v1 = ppo3.estimate_value(np.array([1.0, 0.0], dtype=np.float32))
        print(f"  PPO Probe3: V([0,0])={v0:.3f} (target 0.900), V([1,0])={v1:.3f} (target 1.000)")
        assert abs(v0 - 0.9) < 0.15 and abs(v1 - 1.0) < 0.15, f"PPO Probe3 failed: v0={v0}, v1={v1}"

        # PPO on Probe5 (obs-action coupling): deterministic eval should approach +1.0
        ppo5 = PPOAgent("Probe5-v0", lr=3e-3, max_steps=1500, gamma=0.99, rollout_steps=128, batch_size=64, update_epochs=10, save_interval=10000, seed=0)
        ppo5.train_ppo()
        ev5 = utils.evaluate_policy(ppo5.actor, ppo5.env.observation_space, "Probe5-v0", episodes=20)
        print(f"  PPO Probe5: eval={ev5:.3f} (target > 0.85)")
        assert ev5 > 0.85, f"PPO Probe5 failed: {ev5}"

        # SAC on Probe5 (obs-action coupling): deterministic eval should approach +1.0
        sac5 = SACAgent("Probe5-v0", lr=3e-3, max_steps=600, batch_size=32, start_steps=50, save_interval=10000, seed=0)
        sac5.train_sac()
        ev5_sac = utils.evaluate_policy(sac5.policy, sac5.env.observation_space, "Probe5-v0", episodes=20)
        print(f"  SAC Probe5: eval={ev5_sac:.3f} (target > 0.85)")
        assert ev5_sac > 0.85, f"SAC Probe5 failed: {ev5_sac}"

        # REINFORCE on Probe4 (1-step policy gradient): mean action should approach +1.0
        rf4 = REINFORCEAgent("Probe4-v0", lr=1e-2, max_steps=400, save_interval=10000, seed=0)
        rf4.train_reinforce()
        ev4_rf = utils.evaluate_policy(rf4.policy, rf4.env.observation_space, "Probe4-v0", episodes=10)
        print(f"  REINFORCE Probe4: eval={ev4_rf:.3f} (target > 0.85)")
        assert ev4_rf > 0.85, f"REINFORCE Probe4 failed: {ev4_rf}"

        print("  [PASS] All Andy Jones probe environments passed.")
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_math_units()
    test_probe_environments()
