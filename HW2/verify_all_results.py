#!/usr/bin/env python3
"""
Comprehensive Verification and Audit Script for CS 557 HW2
Author: Tri Vo (vo43@purdue.edu)
Verifies:
1. All TensorBoard event logs (Part 1, Part 2A, Part 2B)
2. Latest saved checkpoints via 20-episode deterministic Gymnasium evaluation
3. Gymnasium reward thresholds (MountainCarContinuous-v0 >= 90.0, BipedalWalker-v3 >= 300.0)
4. Integrity of plots, REPORT.md, Tri_Vo_hw2.pdf, selected_runs/, and tri_vo_hw2.zip
"""

import os
import sys
import glob
import zipfile
import numpy as np
import torch

if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(1)

import gymnasium as gym
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from reinforce import MLPNetwork as ReinforcePolicy
from ppo import ActorNetwork as PPOActor
from sac import PolicyNetwork as SACPolicy
from plot_and_analyze import find_exact_run, load_scalars_from_dir
import utils

RUNS_DIR = os.path.join(BASE_DIR, "runs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def audit_tensorboard_runs():
    print("=" * 95)
    print("1. TENSORBOARD RUNS AUDIT")
    print("=" * 95)

    for env in ["mountaincarcontinuous_v0", "bipedalwalker_v3"]:
        env_dir = os.path.join(RUNS_DIR, env)
        if not os.path.exists(env_dir):
            continue
        for run_name in sorted(os.listdir(env_dir)):
            rpath = os.path.join(env_dir, run_name)
            if not os.path.isdir(rpath):
                continue
            data = load_scalars_from_dir(rpath)
            if "train/episode_reward" not in data:
                continue
            steps, vals = data["train/episode_reward"]
            last10 = vals[-10:] if len(vals) >= 10 else vals
            ev_str = "N/A"
            if "train/eval_mean_reward" in data:
                _, ev_v = data["train/eval_mean_reward"]
                ev_str = f"{ev_v[-1]:>7.2f}"
            print(
                f"{env[:12]:<12} | {run_name[:44]:<44} | Steps: {int(steps[-1]):>6} | "
                f"Last10: {np.mean(last10):>7.2f}±{np.std(last10):>5.2f} | Max: {np.max(vals):>7.2f} | FinalEval: {ev_str}"
            )


def _eval_ckpt(ckpt_path, agent_type, env_name, episodes=10):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device("cpu")
    try:
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
        if agent_type == "reinforce":
            policy = ReinforcePolicy(obs_dim, act_dim).to(device)
            policy.load_state_dict(data)
        elif agent_type == "ppo":
            if isinstance(data, dict) and "actor" in data:
                policy = PPOActor(
                    obs_dim, act_dim,
                    log_std_init=data.get("log_std_init", -0.5),
                    norm_obs=data.get("norm_obs", False),
                    use_sde=data.get("use_sde", False),
                    use_tanh=data.get("use_tanh", False),
                ).to(device)
                policy.load_state_dict(data["actor"])
            else:
                policy = PPOActor(obs_dim, act_dim).to(device)
                policy.load_state_dict(data)
        elif agent_type == "sac":
            policy = SACPolicy(obs_dim, act_dim).to(device)
            policy.load_state_dict(data)

        policy.eval()
        low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        ep_rewards = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            ep_r = 0.0
            while not done:
                obs_vec = torch.from_numpy(utils.encode_obs(obs, env.observation_space)).float().unsqueeze(0)
                with torch.no_grad():
                    raw_a = policy.deterministic_action(obs_vec)
                    scaled_a = low + 0.5 * (raw_a + 1.0) * (high - low)
                act_np = np.clip(scaled_a.cpu().numpy().flatten(), env.action_space.low, env.action_space.high)
                obs, r, term, trunc, _ = env.step(act_np)
                ep_r += float(r)
                done = term or trunc
            ep_rewards.append(ep_r)
        return float(np.mean(ep_rewards)), float(np.std(ep_rewards))
    finally:
        env.close()


def _latest_ckpt_for_run(run_dir, env_name):
    if not run_dir:
        return None, 0
    run_name = os.path.basename(run_dir)
    pattern = os.path.join(CHECKPOINTS_DIR, f"{run_name}_{env_name}_steps_*.pth")
    matched = glob.glob(pattern)
    if not matched:
        return None, 0

    def get_step(p):
        return int(p.split("_steps_")[-1].replace(".pth", ""))

    best = max(matched, key=get_step)
    return best, get_step(best)


def audit_checkpoints():
    print("\n" + "=" * 95)
    print("2. CHECKPOINT DETERMINISTIC EVALUATION AUDIT (20 HELD-OUT EPISODES)")
    print("=" * 95)

    specs = [
        ("Part 1A: REINFORCE", "reinforce", "MountainCarContinuous-v0", "mountaincarcontinuous_v0", {"lr": 5e-5, "gamma": 0.99}, (), None),
        ("Part 1B: PPO (gSDE)", "ppo", "MountainCarContinuous-v0", "mountaincarcontinuous_v0", {"lr": 0.0003, "gamma": 0.999, "sde": 1, "norm": 1}, (), 90.0),
        ("Part 1C: SAC (Bonus)", "sac", "MountainCarContinuous-v0", "mountaincarcontinuous_v0", {"lr": 0.0003, "ou": 1}, (), 90.0),
        ("Part 1A: REINFORCE", "reinforce", "BipedalWalker-v3", "bipedalwalker_v3", {"lr": 5e-5, "gamma": 0.99}, (), None),
        ("Part 1B: PPO Base", "ppo", "BipedalWalker-v3", "bipedalwalker_v3", {"lr": 5e-5, "clip": 0.2, "ent": 0.01, "bs": 256}, ("anneal", "sde", "norm"), None),
        ("Part 1C: SAC (Bonus)", "sac", "BipedalWalker-v3", "bipedalwalker_v3", {"lr": 0.0007, "bs": 256, "tau": 0.01}, (), 300.0),
        ("Part 2B: PPO Tuned", "ppo", "BipedalWalker-v3", "bipedalwalker_v3", {"lr": 0.001, "gamma": 0.995, "clip": 0.2, "ent": 0.0, "bs": 256, "steps": 8192, "anneal": 1}, (), 300.0),
    ]

    all_thresholds_met = True
    for label, agent_type, env_name, env_tag, hparams, forbid, thresh in specs:
        rdir = find_exact_run(env_tag, agent_type, hparams, forbidden_keys=forbid)
        ckpt, step = _latest_ckpt_for_run(rdir, env_name)
        if not ckpt:
            print(f"[{label:<22}] {env_name:<24} | MISSING CHECKPOINT")
            if thresh is not None:
                all_thresholds_met = False
            continue
        mean_r, std_r = _eval_ckpt(ckpt, agent_type, env_name, episodes=20)
        status = ""
        if thresh is not None:
            passed = mean_r >= thresh
            status = f" | Threshold >= {thresh:.0f}: {'PASS' if passed else 'FAIL'}"
            if not passed:
                all_thresholds_met = False
        print(f"[{label:<22}] {env_name:<24} | Step: {step:>7} | Eval(20): {mean_r:>7.2f} ± {std_r:>5.2f}{status}")

    print(f"\nAll Success Thresholds Met: {all_thresholds_met}")
    return all_thresholds_met


def audit_submission_artifacts():
    print("\n" + "=" * 95)
    print("3. SUBMISSION ARTIFACTS INTEGRITY AUDIT")
    print("=" * 95)

    pdf_path = os.path.join(BASE_DIR, "Tri_Vo_hw2.pdf")
    zip_path = os.path.join(BASE_DIR, "tri_vo_hw2.zip")
    report_path = os.path.join(BASE_DIR, "REPORT.md")

    all_ok = True
    for name, path in [
        ("Tri_Vo_hw2.pdf", pdf_path),
        ("REPORT.md", report_path),
        ("tri_vo_hw2.zip", zip_path),
        ("plots/metrics_summary.json", os.path.join(PLOTS_DIR, "metrics_summary.json")),
    ]:
        if os.path.exists(path):
            print(f"[OK] {name} exists ({os.path.getsize(path) / 1024:.1f} KB)")
        else:
            print(f"[MISSING] {name}")
            all_ok = False

    expected_plots = [
        "part1_mountaincar_reward.png",
        "part1_bipedal_reward.png",
        "part1_reinforce_losses.png",
        "part1_ppo_losses.png",
        "bonus_sac_mountaincar_diagnostics.png",
        "bonus_sac_bipedal_diagnostics.png",
        "part2a_clip_ratio_ablation.png",
        "part2a_entropy_coef_ablation.png",
        "part2a_lr_ablation.png",
        "part2a_batch_size_ablation.png",
        "part2b_best_vs_baseline.png",
    ]
    for ep in expected_plots:
        p = os.path.join(PLOTS_DIR, ep)
        if os.path.exists(p):
            print(f"  [OK] plot: {ep} ({os.path.getsize(p) / 1024:.1f} KB)")
        else:
            print(f"  [MISSING] plot: {ep}")
            all_ok = False

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            required = [
                "reinforce.py", "ppo.py", "sac.py", "main.py", "utils.py", "logger.py",
                "Tri_Vo_hw2.pdf", "REPORT.md", "submit_all_jobs.sh", "test_agents.py",
            ]
            for rf in required:
                present = rf in names
                print(f"  zip [{'OK' if present else 'MISSING'}]: {rf}")
                if not present:
                    all_ok = False
            sel = [n for n in names if n.startswith("selected_runs/") and "events.out.tfevents" in n]
            print(f"  zip selected_runs event files: {len(sel)}")
            if len(sel) != 7:
                all_ok = False
    print("=" * 95)
    return all_ok


if __name__ == "__main__":
    audit_tensorboard_runs()
    thresholds_ok = audit_checkpoints()
    artifacts_ok = audit_submission_artifacts()
    if not (thresholds_ok and artifacts_ok):
        sys.exit(1)
