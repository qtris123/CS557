#!/usr/bin/env python3
"""
Publication-Quality Analysis & Plotting for CS 557 HW2
Author: Tri Vo (vo43@purdue.edu)
Extracts TensorBoard scalar summaries with exact hyperparameter matching (no prefix collisions)
and produces colorblind-accessible Matplotlib figures + verified metrics_summary.json.
"""

import json
import os
import glob
import re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Okabe-Ito colorblind-safe qualitative palette + redundant linestyles
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]

plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 13.5,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_scalars_from_dir(log_dir):
    """Load all scalar curves from the latest TensorBoard event file in log_dir."""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return {}
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        values = np.array([e.value for e in events], dtype=np.float64)
        scalars[tag] = (steps, values)
    return scalars


def parse_run_dir_hparams(dirname):
    """
    Parse source tag and hyperparameter tokens from Logger directory name.
    Example: ppo_lr0.0003_gamma0.99_lam0.95_clip0.2_ent0.01_bs128_epochs10_steps2048_lsi-0.5_1795132395
    """
    parts = dirname.split("_")
    source = parts[0]
    parsed = {"_source": source}
    # Strip trailing timestamp (and optional collision suffix)
    token_parts = parts[1:]
    while token_parts and token_parts[-1].isdigit():
        token_parts.pop()
    for tok in token_parts:
        m = re.match(r"^([a-zA-Z]+)(-?\d.*)$", tok)
        if m:
            k, v = m.group(1), m.group(2)
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
    return parsed


def find_exact_run(env_tag, source_tag, required_hparams, forbidden_keys=()):
    """
    Find the latest run directory under runs/<env_tag> matching source_tag and ALL
    required_hparams (exact numeric match), excluding runs with forbidden_keys.
    """
    env_dir = os.path.join(RUNS_DIR, env_tag)
    if not os.path.exists(env_dir):
        return None
    matches = []
    for name in sorted(os.listdir(env_dir)):
        full_path = os.path.join(env_dir, name)
        if not os.path.isdir(full_path):
            continue
        parsed = parse_run_dir_hparams(name)
        if parsed.get("_source") != source_tag:
            continue
        if any(fk in parsed for fk in forbidden_keys):
            continue
        ok = True
        for k, v in required_hparams.items():
            if k not in parsed:
                ok = False
                break
            if isinstance(v, (int, float)) and isinstance(parsed[k], (int, float)):
                if not np.isclose(parsed[k], float(v), rtol=1e-6, atol=1e-12):
                    ok = False
                    break
            elif str(parsed[k]) != str(v):
                ok = False
                break
        if ok:
            matches.append(full_path)

    if not matches:
        return None
    # Prefer the run with the most environment steps (ties broken by latest mtime)
    def score_run(rdir):
        sc = load_scalars_from_dir(rdir)
        rew = sc.get("train/episode_reward")
        max_step = float(rew[0][-1]) if (rew is not None and len(rew[0]) > 0) else 0.0
        return (max_step, os.path.getmtime(rdir))

    best = max(matches, key=score_run)
    print(f"  Matched [{env_tag} | {source_tag} | {required_hparams}] -> {os.path.basename(best)}")
    return best


def smooth(values, weight=0.85):
    """Exponential moving average for smoothing noisy RL curves."""
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values, dtype=np.float64)
    last = float(values[0])
    for i, val in enumerate(values):
        last = last * weight + (1.0 - weight) * float(val)
        smoothed[i] = last
    return smoothed


def summarize_run(data):
    """Extract ground-truth quantitative summary from loaded scalar dict."""
    out = {}
    if "train/episode_reward" in data:
        steps, vals = data["train/episode_reward"]
        last10 = vals[-10:] if len(vals) >= 10 else vals
        out["episodes"] = int(len(vals))
        out["final_step"] = int(steps[-1])
        out["mean_last10"] = float(np.mean(last10))
        out["std_last10"] = float(np.std(last10))
        out["max_train_reward"] = float(np.max(vals))
        out["final_train_reward"] = float(vals[-1])
    if "train/eval_mean_reward" in data:
        ev_s, ev_v = data["train/eval_mean_reward"]
        out["eval_history"] = [(int(s), float(v)) for s, v in zip(ev_s, ev_v)]
        out["final_eval_reward"] = float(ev_v[-1])
        out["max_eval_reward"] = float(np.max(ev_v))
    return out


def plot_comparison(runs_dict, title, ylabel, save_path, threshold=None, threshold_label=None, ylim=None):
    """Plot a set of runs with raw semi-transparent traces, EMA smoothing, and optional threshold."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout="constrained", facecolor="white")

    for i, (label, (steps, values)) in enumerate(runs_dict.items()):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        if len(steps) > 10:
            smoothed_vals = smooth(values, weight=0.82)
            ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(steps, smoothed_vals, color=color, linestyle=ls, label=label, linewidth=2.1)
        else:
            ax.plot(steps, values, color=color, linestyle=ls, marker="o", markersize=4, label=label, linewidth=2.0)

    if threshold is not None:
        ax.axhline(
            threshold,
            color="#333333",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            label=threshold_label or f"Solved Threshold ({threshold:g})",
        )

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.92)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_reinforce_losses(mc_data, bw_data, save_path):
    """Plot REINFORCE policy loss curves for both environments (required by Part 1A)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), layout="constrained", facecolor="white")
    panels = [
        (axes[0], mc_data, "MountainCarContinuous-v0: REINFORCE Policy Loss"),
        (axes[1], bw_data, "BipedalWalker-v3: REINFORCE Policy Loss"),
    ]
    for ax, data, title in panels:
        if data and "train/policy_loss" in data:
            s, v = data["train/policy_loss"]
            ax.plot(s, v, color=OKABE_ITO[0], alpha=0.22, linewidth=0.8, label="Raw Loss")
            ax.plot(s, smooth(v, 0.85), color=OKABE_ITO[0], linewidth=2.0, label="EMA (0.85)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Policy Loss")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_ppo_losses(mc_data, bw_data, save_path):
    """Plot PPO actor and critic loss curves for both Part 1B environments."""
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), layout="constrained", facecolor="white")
    specs = [
        (axes[0, 0], mc_data, "train/actor_loss", "MountainCarContinuous-v0: PPO Actor Loss", "Actor Loss", OKABE_ITO[0]),
        (axes[0, 1], mc_data, "train/critic_loss", "MountainCarContinuous-v0: PPO Critic Loss", "Critic Loss (MSE)", OKABE_ITO[1]),
        (axes[1, 0], bw_data, "train/actor_loss", "BipedalWalker-v3: PPO Actor Loss", "Actor Loss", OKABE_ITO[0]),
        (axes[1, 1], bw_data, "train/critic_loss", "BipedalWalker-v3: PPO Critic Loss", "Critic Loss (MSE)", OKABE_ITO[1]),
    ]
    for ax, data, tag, title, ylabel, color in specs:
        if data and tag in data:
            s, v = data[tag]
            ax.plot(s, v, color=color, alpha=0.22, linewidth=0.8, label="Raw")
            ax.plot(s, smooth(v, 0.82), color=color, linewidth=2.0, label="EMA (0.82)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_sac_diagnostics(data, env_name, threshold, save_path):
    """Dual-panel SAC diagnostics: Episode Reward (+ threshold) and Critic/Policy Losses + Alpha."""
    if not data:
        print(f"Missing SAC data for {env_name}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6), layout="constrained", facecolor="white")

    # Panel 1: Reward + Deterministic Eval
    if "train/episode_reward" in data:
        steps, rewards = data["train/episode_reward"]
        ax1.plot(steps, rewards, color=OKABE_ITO[0], alpha=0.22, linewidth=0.8, label="Train Episode Return (Raw)")
        ax1.plot(steps, smooth(rewards, 0.82), color=OKABE_ITO[0], linewidth=2.1, label="Train Return (EMA 0.82)")
    if "train/eval_mean_reward" in data:
        es, ev = data["train/eval_mean_reward"]
        ax1.plot(es, ev, color=OKABE_ITO[2], marker="s", markersize=4.5, linewidth=1.6, linestyle="--", label="Deterministic Eval (10 eps)")
    if threshold is not None:
        ax1.axhline(threshold, color="#333333", linestyle=":", linewidth=1.4, label=f"Solved Threshold ({threshold:g})")
    ax1.set_title(f"{env_name}: SAC Episode & Eval Return", fontweight="bold")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Return")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="best", framealpha=0.9)

    # Panel 2: Critic Loss, Policy Loss, and Temperature Alpha
    lines, labels = [], []
    if "train/critic_loss" in data:
        sc, lc = data["train/critic_loss"]
        l1, = ax2.plot(sc, smooth(lc, 0.85), color=OKABE_ITO[1], linewidth=1.9, label="Critic Loss (MSBE)")
        lines.append(l1); labels.append("Critic Loss (MSBE)")
    if "train/policy_loss" in data:
        sp, lp = data["train/policy_loss"]
        l2, = ax2.plot(sp, smooth(lp, 0.85), color=OKABE_ITO[2], linestyle="--", linewidth=1.9, label="Policy Loss")
        lines.append(l2); labels.append("Policy Loss")
    if "train/alpha" in data:
        sa, la = data["train/alpha"]
        ax2_twin = ax2.twinx()
        ax2_twin.spines["top"].set_visible(False)
        l3, = ax2_twin.plot(sa, la, color=OKABE_ITO[3], linestyle=":", linewidth=2.0, label="Temperature alpha")
        ax2_twin.set_ylabel("Temperature alpha", color=OKABE_ITO[3])
        ax2_twin.tick_params(axis="y", labelcolor=OKABE_ITO[3])
        lines.append(l3); labels.append("Temperature alpha")

    ax2.set_title(f"{env_name}: SAC Losses & Temperature", fontweight="bold")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Loss Magnitude")
    ax2.grid(True, linestyle="--", alpha=0.3)
    if lines:
        ax2.legend(lines, labels, loc="best", framealpha=0.9)

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved SAC diagnostic plot: {save_path}")


def analyze_and_plot():
    print("Gathering runs with exact hyperparameter matching...")
    summary = {}

    # --- Part 1: MountainCarContinuous-v0 ---
    rf_mc_dir = find_exact_run("mountaincarcontinuous_v0", "reinforce", {"lr": 5e-5, "gamma": 0.99})
    ppo_mc_dir = find_exact_run("mountaincarcontinuous_v0", "ppo", {"lr": 0.0003, "gamma": 0.999, "sde": 1, "norm": 1})
    sac_mc_dir = find_exact_run("mountaincarcontinuous_v0", "sac", {"lr": 0.0003, "ou": 1})

    rf_mc = load_scalars_from_dir(rf_mc_dir) if rf_mc_dir else {}
    ppo_mc = load_scalars_from_dir(ppo_mc_dir) if ppo_mc_dir else {}
    sac_mc = load_scalars_from_dir(sac_mc_dir) if sac_mc_dir else {}

    mc_runs = {}
    if "train/episode_reward" in rf_mc:
        mc_runs["REINFORCE (Part 1A)"] = rf_mc["train/episode_reward"]
        summary["p1_reinforce_mc"] = summarize_run(rf_mc)
    if "train/episode_reward" in ppo_mc:
        mc_runs["PPO + gSDE (Part 1B)"] = ppo_mc["train/episode_reward"]
        summary["p1_ppo_mc"] = summarize_run(ppo_mc)
    if "train/episode_reward" in sac_mc:
        mc_runs["SAC + OU Warmup (Part 1C Bonus)"] = sac_mc["train/episode_reward"]
        summary["p1_sac_mc"] = summarize_run(sac_mc)

    if mc_runs:
        plot_comparison(
            mc_runs,
            "Part 1: MountainCarContinuous-v0 Episode Return",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part1_mountaincar_reward.png"),
            threshold=90.0,
            threshold_label="Gymnasium Threshold (+90.0)",
        )

    # --- Part 1: BipedalWalker-v3 ---
    rf_bw_dir = find_exact_run("bipedalwalker_v3", "reinforce", {"lr": 5e-5, "gamma": 0.99})
    ppo_bw_dir = find_exact_run(
        "bipedalwalker_v3", "ppo",
        {"lr": 5e-5, "clip": 0.2, "ent": 0.01, "bs": 256},
        forbidden_keys=("anneal", "sde", "norm"),
    )
    sac_bw_dir = find_exact_run("bipedalwalker_v3", "sac", {"lr": 0.0007, "bs": 256, "tau": 0.01})

    rf_bw = load_scalars_from_dir(rf_bw_dir) if rf_bw_dir else {}
    ppo_bw = load_scalars_from_dir(ppo_bw_dir) if ppo_bw_dir else {}
    sac_bw = load_scalars_from_dir(sac_bw_dir) if sac_bw_dir else {}

    bw_runs = {}
    if "train/episode_reward" in rf_bw:
        bw_runs["REINFORCE (Part 1A)"] = rf_bw["train/episode_reward"]
        summary["p1_reinforce_bipedal"] = summarize_run(rf_bw)
    if "train/episode_reward" in ppo_bw:
        bw_runs["PPO Baseline (Part 1B)"] = ppo_bw["train/episode_reward"]
        summary["p1_ppo_bipedal"] = summarize_run(ppo_bw)
    if "train/episode_reward" in sac_bw:
        bw_runs["SAC (Part 1C Bonus)"] = sac_bw["train/episode_reward"]
        summary["p1_sac_bipedal"] = summarize_run(sac_bw)

    if bw_runs:
        plot_comparison(
            bw_runs,
            "Part 1: BipedalWalker-v3 Episode Return",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part1_bipedal_reward.png"),
            threshold=300.0,
            threshold_label="Gymnasium Threshold (+300.0)",
        )

    # --- Part 1A & 1B Loss Curves ---
    plot_reinforce_losses(rf_mc, rf_bw, os.path.join(PLOTS_DIR, "part1_reinforce_losses.png"))
    plot_ppo_losses(ppo_mc, ppo_bw, os.path.join(PLOTS_DIR, "part1_ppo_losses.png"))

    # --- Part 1C Bonus: SAC Diagnostics ---
    plot_sac_diagnostics(sac_mc, "MountainCarContinuous-v0", 90.0, os.path.join(PLOTS_DIR, "bonus_sac_mountaincar_diagnostics.png"))
    plot_sac_diagnostics(sac_bw, "BipedalWalker-v3", 300.0, os.path.join(PLOTS_DIR, "bonus_sac_bipedal_diagnostics.png"))

    # --- Part 2A: PPO BipedalWalker-v3 Ablations ---
    # 1. Clip Ratio (0.1, 0.2, 0.3) at lr=5e-5, ent=0.01, bs=256
    clip_runs = {}
    for clip_val in [0.1, 0.2, 0.3]:
        rdir = find_exact_run(
            "bipedalwalker_v3", "ppo",
            {"lr": 5e-5, "clip": clip_val, "ent": 0.01, "bs": 256},
            forbidden_keys=("anneal", "sde", "norm"),
        )
        if rdir:
            d = load_scalars_from_dir(rdir)
            if "train/episode_reward" in d:
                clip_runs[f"clip={clip_val}"] = d["train/episode_reward"]
                summary[f"p2a_clip_{clip_val}"] = summarize_run(d)
    if len(clip_runs) > 1:
        plot_comparison(
            clip_runs,
            "Part 2A: PPO Clip Ratio Ablation on BipedalWalker-v3 (lr=5e-5, bs=256)",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part2a_clip_ratio_ablation.png"),
        )

    # 2. Entropy Coef (0.0, 0.01, 0.05) at lr=5e-5, clip=0.2, bs=256
    ent_runs = {}
    for ent_val in [0.0, 0.01, 0.05]:
        rdir = find_exact_run(
            "bipedalwalker_v3", "ppo",
            {"lr": 5e-5, "clip": 0.2, "ent": ent_val, "bs": 256},
            forbidden_keys=("anneal", "sde", "norm"),
        )
        if rdir:
            d = load_scalars_from_dir(rdir)
            if "train/episode_reward" in d:
                ent_runs[f"ent={ent_val}"] = d["train/episode_reward"]
                summary[f"p2a_ent_{ent_val}"] = summarize_run(d)
    if len(ent_runs) > 1:
        plot_comparison(
            ent_runs,
            "Part 2A: PPO Entropy Coefficient Ablation on BipedalWalker-v3 (lr=5e-5, bs=256)",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part2a_entropy_coef_ablation.png"),
        )

    # 3. Learning Rate (1e-5, 5e-5, 3e-4, 1e-3) at clip=0.2, ent=0.01, bs=256
    lr_runs = {}
    for lr_val in [1e-5, 5e-5, 3e-4, 1e-3]:
        rdir = find_exact_run(
            "bipedalwalker_v3", "ppo",
            {"lr": lr_val, "clip": 0.2, "ent": 0.01, "bs": 256},
            forbidden_keys=("anneal", "sde", "norm"),
        )
        if rdir:
            d = load_scalars_from_dir(rdir)
            if "train/episode_reward" in d:
                lr_runs[f"lr={lr_val:g}"] = d["train/episode_reward"]
                summary[f"p2a_lr_{lr_val:g}"] = summarize_run(d)
    if len(lr_runs) > 1:
        plot_comparison(
            lr_runs,
            "Part 2A: PPO Learning Rate Ablation on BipedalWalker-v3 (bs=256, clip=0.2)",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part2a_lr_ablation.png"),
        )

    # 4. Batch Size (64, 128, 256) at lr=3e-4, clip=0.2, ent=0.01
    bs_runs = {}
    for bs_val in [64, 128, 256]:
        rdir = find_exact_run(
            "bipedalwalker_v3", "ppo",
            {"lr": 0.0003, "clip": 0.2, "ent": 0.01, "bs": bs_val},
            forbidden_keys=("anneal", "sde", "norm"),
        )
        if rdir:
            d = load_scalars_from_dir(rdir)
            if "train/episode_reward" in d:
                bs_runs[f"bs={bs_val}"] = d["train/episode_reward"]
                summary[f"p2a_bs_{bs_val}"] = summarize_run(d)
    if len(bs_runs) > 1:
        plot_comparison(
            bs_runs,
            "Part 2A: PPO Mini-Batch Size Ablation on BipedalWalker-v3 (lr=3e-4, clip=0.2)",
            "Episode Return",
            os.path.join(PLOTS_DIR, "part2a_batch_size_ablation.png"),
        )

    # --- Part 2B: Tuned PPO (rollout_steps=8192, anneal_lr) vs Part 1 Baseline PPO ---
    best_dir = find_exact_run(
        "bipedalwalker_v3", "ppo",
        {"lr": 0.001, "gamma": 0.995, "clip": 0.2, "ent": 0.0, "bs": 256, "steps": 8192, "anneal": 1},
    )
    if ppo_bw and best_dir:
        best_data = load_scalars_from_dir(best_dir)
        if "train/episode_reward" in best_data:
            summary["p2b_ppo_best"] = summarize_run(best_data)
            best_runs = {
                "PPO Baseline (Part 1B: lr=5e-5, bs=256)": ppo_bw["train/episode_reward"],
                "PPO Tuned (Part 2B: lr=1e-3->0, steps=8192)": best_data["train/episode_reward"],
            }
            plot_comparison(
                best_runs,
                "Part 2B: Tuned PPO vs Part 1 Baseline on BipedalWalker-v3",
                "Episode Return",
                os.path.join(PLOTS_DIR, "part2b_best_vs_baseline.png"),
                threshold=300.0,
                threshold_label="Gymnasium Threshold (+300.0)",
            )

    summary_path = os.path.join(PLOTS_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary: {summary_path}")


if __name__ == "__main__":
    analyze_and_plot()
