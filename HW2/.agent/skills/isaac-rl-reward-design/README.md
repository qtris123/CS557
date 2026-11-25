# Isaac RL Reward Design

A reusable Agent Skill for designing, reviewing, debugging, and tuning reinforcement-learning rewards for robotics tasks in NVIDIA Isaac Lab / Isaac Gym.

The project focuses on one principle:

> Optimize for the desired physical behavior, not for a large reward number.

## What this skill does

- Designs a minimal reward from a task description.
- Reviews an existing `RewardsCfg` term by term.
- Diagnoses reward-scale imbalance from episodic reward breakdowns.
- Detects common reward-hacking patterns.
- Separates task reward, shaping, stability, smoothness, energy, contact, safety, and termination terms.
- Suggests curriculum schedules instead of enabling every penalty at full strength from step 0.
- Produces controlled ablation experiments before recommending PPO hyperparameter changes.

## Repository layout

```text
isaac-rl-reward-design/
├── SKILL.md
├── README.md
├── references/
│   ├── locomotion.md
│   ├── manipulation.md
│   └── reward-debugging.md
└── scripts/
    └── analyze_rewards.py
```

`SKILL.md` contains the decision process. The files under `references/` are loaded only when a task needs that domain, keeping the core skill smaller and easier to evolve.

## Core workflow

```text
Desired behavior
      ↓
Minimal task reward
      ↓
Policy behavior
      ↓
Per-term reward breakdown
      ↓
Exploit / conflict diagnosis
      ↓
Simplify or rebalance
      ↓
Controlled retraining
```

## Quick use

Give the agent one or more of the following:

- robot type and task objective
- current `RewardsCfg`
- termination conditions
- command ranges and control frequency
- TensorBoard or CSV reward breakdown
- a video, screenshots, or a behavior description

Then ask it to design or review the reward.

Example prompts:

```text
Review this Isaac Lab RewardsCfg. Identify conflicting terms, scale risks,
reward hacking opportunities, and propose the smallest corrected version.
```

```text
The quadruped tracks velocity but refuses commands above 1.5 m/s.
Use the reward breakdown below to diagnose whether energy/stability penalties
are making fast motion economically unattractive.
```

```text
Design a staged pick-and-place reward for this manipulation task.
Use gating so that reach, grasp, lift, transport, and placement objectives do
not all compete from the first timestep.
```

## Analyze reward contributions

`scripts/analyze_rewards.py` uses only the Python standard library.

For a CSV containing weighted reward columns:

```bash
python scripts/analyze_rewards.py rewards.csv
```

Select raw term columns explicitly:

```bash
python scripts/analyze_rewards.py rewards.csv \
  --columns track_lin_vel track_ang_vel torque action_rate
```

If rows are timesteps and the file has an episode id, aggregate each term per episode first:

```bash
python scripts/analyze_rewards.py rewards.csv --episode-column episode_id
```

For raw reward functions, weights and `dt` can be applied during analysis:

```bash
python scripts/analyze_rewards.py rewards.csv \
  --columns track_lin_vel torque \
  --weights-json '{"track_lin_vel": 2.0, "torque": -1e-5}' \
  --dt 0.02
```

The output ranks terms by absolute contribution and reports mean, standard deviation, p95 absolute magnitude, max absolute magnitude, total contribution, and positive/negative budget share.

The script is intentionally a diagnostic tool, not an automatic weight tuner.

## Reward design philosophy

1. Task completion comes before elegance.
2. Primary task reward should remain visible in the reward budget.
3. Judge weighted contributions, not weights in isolation.
4. Add regularization only to fix observed behavior.
5. Every shaping reward must be checked for an unintended shortcut.
6. A term that does not affect behavior should probably be removed.
7. Change as few variables as possible between experiments.

## Isaac Lab notes

For version-specific API details, use your installed Isaac Lab source or the current official documentation. In current Isaac Lab, `RewardManager` applies the environment timestep to reward weights, `mdp.is_terminated` excludes normal timeouts, and curriculum utilities can modify reward weights during training.

Official docs:

- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html
- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html

## Status

Initial version. The first release targets locomotion and manipulation tasks and provides a generic reward-debugging workflow that can be extended to humanoids, wheel-legged robots, drones, dexterous hands, and custom Isaac Lab tasks.
