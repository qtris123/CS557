---
name: isaac-rl-reward-design
description: >
  Design, review, debug, and tune reinforcement-learning reward functions for
  NVIDIA Isaac Lab / Isaac Gym robotics tasks. Use for RewardsCfg review,
  locomotion, manipulation, humanoids, quadrupeds, wheel-legged robots, drones,
  reward hacking, standing-still policies, unstable learning, excessive
  penalties, jerky motion, poor tracking, or reward-scale imbalance.
---

# Isaac RL Reward Design

Design the smallest reward function that reliably produces the intended physical
behavior.

Do not optimize for the number of reward terms or for total reward alone.
Optimize for:

- task performance
- learnability
- stability
- smoothness
- physical plausibility
- robustness
- interpretability

## Source of truth

When exact Isaac Lab API behavior matters, prefer the installed/version-matched
Isaac Lab source or current official documentation over memory.

Important current behaviors to remember:

- `RewardManager` computes a weighted sum of reward terms.
- Isaac Lab applies the environment time-step `dt` to reward-term weights.
- episodic per-term reward sums are more informative than total reward alone.
- `mdp.is_terminated` excludes normal episodic timeouts from its termination
  penalty.
- reward weights can be changed during training with curriculum utilities.

Official references:

- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html
- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html

## Progressive references

Load only the reference needed for the task:

- locomotion, humanoid, quadruped, wheel-legged: `references/locomotion.md`
- grasping, pick-and-place, object manipulation: `references/manipulation.md`
- strange behavior, reward breakdown, training failure: `references/reward-debugging.md`

If a task spans multiple categories, load only the relevant sections.

---

# Core model

Treat total reward as:

```text
r_t = sum_i weight_i * term_i(state, action) * dt
```

Do not judge a numerical weight in isolation.

A weight such as:

```python
weight = -0.01
```

has no useful meaning until the raw term magnitude is known.

For example:

```text
action_rate raw ~= 2
joint_acc raw   ~= 100000
torque raw      ~= 500
```

Always reason about:

```text
weighted contribution = raw value * weight * dt
```

When data is available, inspect for each term:

- mean
- standard deviation
- p95
- max absolute value
- episodic sum
- share of positive reward
- share of negative reward

Use `scripts/analyze_rewards.py` for CSV summaries when applicable.

---

# Reward hierarchy

Classify every term into exactly one primary category before tuning it:

```text
PRIMARY
SHAPING
STABILITY
SMOOTHNESS
ENERGY
CONTACT
SAFETY
TERMINATION
STYLE
METRIC_ONLY
```

## Primary task reward

This is what the robot is fundamentally supposed to accomplish.

Examples:

- linear velocity tracking
- yaw-rate tracking
- target position tracking
- trajectory tracking
- end-effector pose tracking
- grasp success
- lift success
- object placement

Primary task terms should normally dominate the useful reward budget.

If auxiliary penalties make task progress economically unattractive, reduce,
gate, or delay those penalties before changing PPO hyperparameters.

## Shaping reward

Use shaping only to expose progress toward a sparse goal.

Ask:

> Does this term tell the policy whether it is closer to success?

Typical bounded tracking form:

```text
exp(-error^2 / sigma^2)
```

Use bounded tracking rewards when exact tracking is desired and very large
errors should not dominate optimization.

Every shaping term must be checked for a shortcut that produces reward without
making real task progress.

## Regularization

Regularization changes how the robot performs an already learnable task.

Typical terms:

- action rate
- action magnitude
- joint acceleration
- torque
- power / energy
- base vertical velocity
- roll/pitch angular velocity
- joint limits
- foot slip
- undesired contacts

Add regularization to fix observed failure modes. Do not add it merely because a
reference environment has the term.

## Termination

Separate true failure from normal timeout.

True failures can include:

- falling
- forbidden body-ground contact
- object drop
- leaving workspace
- unsafe collision

Do not penalize normal successful timeout as if it were a catastrophic failure.

---

# Default workflow

Follow this order unless task structure strongly requires otherwise.

## Step 0 — Verify environment semantics

Before reward tuning, check obvious non-reward causes:

1. observation correctness
2. command generation and command ranges
3. action scaling
4. control frequency / decimation
5. actuator configuration
6. contact sensor configuration
7. termination conditions
8. reward implementation

Do not use reward weights to hide a broken environment.

## Step 1 — Define desired behavior

Write the desired physical behavior in one sentence.

Examples:

```text
Track commanded planar velocity while remaining upright.
```

```text
Grasp the cube, lift it above 15 cm, move it to the target, and release it.
```

If the behavior cannot be stated clearly, the reward cannot be evaluated
clearly.

## Step 2 — Build the minimum viable reward

Start with only the primary objective and required failure handling.

Ask:

```text
Can the policy learn the basic task?
```

Do not begin with 20–30 reward terms.

## Step 3 — Observe behavior

Use video and rollout behavior as first-class debugging signals.

Record:

- what the robot does well
- what it refuses to do
- which motion looks unstable
- which shortcuts appear
- when failure occurs

## Step 4 — Inspect reward breakdown

Compare actual weighted episodic contributions.

Flag:

- an auxiliary term dominating total negative return
- a reward term that is numerically near zero throughout training
- multiple terms measuring almost the same behavior
- terms whose contribution changes sharply at thresholds
- terms whose sign or scale disagrees with intended behavior

## Step 5 — Add only justified regularization

Examples:

```text
jerky motion       -> inspect action rate / action scaling first
high energy        -> add or strengthen energy regularization later
body wobble        -> inspect orientation/body angular velocity
foot skating       -> add slip/contact terms if needed
```

## Step 6 — Use curriculum when needed

If a penalty prevents initial learning, keep it weak early and strengthen it
after task competence emerges.

Conceptually:

```text
early:
  task reward        strong
  regularization     weak

later:
  task reward        stable
  regularization     stronger
```

Prefer explicit curriculum over making the initial reward unnecessarily harsh.

## Step 7 — Run controlled ablations

Change as few variables as possible between runs.

Good experiment plan:

```text
A: baseline
B: remove dominant penalty
C: restore penalty with reduced weight
D: restore via curriculum
```

Do not simultaneously change reward weights, observations, action scaling, PPO
learning rate, entropy coefficient, and network architecture.

---

# Mandatory diagnostics

Perform these tests when reviewing a reward.

## Zero-action policy

Ask:

```text
What return does the robot get if it does nothing?
```

If doing nothing is competitive with attempting the task, the reward budget is
misaligned.

## Random policy

Ask:

```text
Can violent random behavior accidentally earn meaningful positive reward?
```

If yes, locate the exploitable term.

## Successful trajectory

Ask:

```text
Does the trajectory humans consider best actually receive the best return?
```

If not, reward and intent are misaligned.

## Reward breakdown

Example:

```text
track_lin_vel       +18.0
track_ang_vel        +7.0
orientation          -2.0
action_rate          -3.0
joint_acc            -1.0
torque               -0.2
undesired_contact   -40.0
```

Diagnosis should start with the dominant `undesired_contact` term, not PPO
hyperparameters.

Inspect its:

- raw magnitude
- threshold
- sensor/body selection
- weight
- activation conditions

## Remove-one-term ablation

Remove one term and evaluate behavior.

If behavior does not meaningfully worsen, consider deleting the term.

---

# Reward hacking

For every positive shaping reward, ask:

> How can the policy earn this without doing what the human intends?

Common patterns:

## Proximity hacking

The end effector stays close to an object forever because reach reward remains
high without grasping.

## Lift hacking

The robot strikes or throws an object upward to collect height reward without a
valid grasp.

## Standing-still hacking

Energy, torque, acceleration, or contact penalties make movement more expensive
than tracking reward makes it valuable.

## Threshold oscillation

The policy repeatedly crosses a threshold to retrigger bonuses.

## Contact farming

Repeated contacts generate reward even though they do not improve task state.

## Episode-length hacking

The policy manipulates termination or survival incentives rather than task
progress.

Whenever behavior is strange but return is high, assume reward exploitation
until disproven.

---

# Reward budget

Think in relative contribution, not universal weights.

A conceptual starting budget for a successful trajectory can be:

```text
primary task       1.00
secondary task     0.20 - 0.50
stability          0.05 - 0.20
smoothness         0.01 - 0.10
energy             0.01 - 0.10
```

These are not literal Isaac Lab weights.

Use them only as a sanity model for relative influence.

The primary objective should remain clearly visible after all penalties are
combined.

---

# Review an existing RewardsCfg

For each reward term produce:

```text
Name:
Function:
Weight:
Category:
Intended behavior:
Observed/expected raw scale:
Weighted scale risk:
Possible conflict:
Reward-hacking risk:
Keep / Modify / Remove:
Recommended change:
```

Then inspect term interactions.

Prioritize:

- direct conflicts
- redundant measurements
- hidden double penalties
- zero-command behavior
- contact terms that fire constantly
- penalties whose raw scale is orders of magnitude larger
- timestep-sensitive custom rewards
- sparse rewards with no useful exploration gradient

---

# Design from scratch

When asked to design a reward from a task description, output:

## Task objective

One sentence describing desired physical behavior.

## Minimum viable reward

Only terms needed to learn the basic task.

## Failure conditions

True termination conditions, clearly separated from timeout.

## Optional regularization

Terms to introduce only after observing a need.

## Reward table

Use:

| Term | Category | Function/formula | Role | Main risk |
|---|---|---|---|---|

## Suggested RewardsCfg

Provide Isaac Lab-compatible code when the environment context is sufficient.
Treat weights as starting hypotheses, not universal values.

## Curriculum

State which terms should be delayed or strengthened later.

## Reward-hacking risks

List concrete shortcuts the policy may exploit.

## Metrics to log

Always include per-term episodic contribution.

---

# Diagnose training behavior

When the robot refuses to move, inspect:

- task reward scale
- torque penalty
- action penalty
- action-rate penalty
- joint-acceleration penalty
- contact penalties
- command distribution

Likely pattern:

```text
cost of movement > value of task progress
```

When the robot moves but shakes, inspect:

- action scaling
- control rate / decimation
- actuator model
- action-rate penalty
- joint acceleration

Do not blindly increase smoothness penalties before checking control semantics.

When the robot tracks but falls, inspect:

- command difficulty
- orientation / stability terms
- termination
- base/body constraints
- curriculum

When the robot moves too slowly, inspect whether:

- energy
- torque
- acceleration
- contact
- stability

penalties make high-speed motion economically unattractive.

---

# Debugging priority

Use this order:

```text
1. Environment correctness
2. Observation correctness
3. Action scaling and control semantics
4. Reward implementation correctness
5. Termination correctness
6. Reward scale and interactions
7. Curriculum
8. PPO hyperparameters
```

Do not immediately blame learning rate, entropy, clip range, or network size when
the reward is inconsistent.

---

# Information that improves diagnosis

Use whatever is available:

- robot type
- task objective
- Isaac Lab environment/version
- `RewardsCfg`
- custom reward functions
- termination config
- episode length
- command ranges
- control frequency and decimation
- action representation/scaling
- PPO config
- per-term TensorBoard values
- rollout video or behavior description

Do not block analysis because minor details are missing. Make reasonable
assumptions and label them.

---

# Output quality bar

Prefer a small number of high-confidence changes over a long list of speculative
ones.

When recommending an experiment, state:

1. hypothesis
2. exact reward/config change
3. expected behavioral change
4. metric that would confirm or reject the hypothesis

Never optimize reward numbers in isolation.

Use this loop:

```text
Desired behavior
      ↓
Reward function
      ↓
Policy behavior
      ↓
Reward breakdown
      ↓
Exploit / conflict diagnosis
      ↓
Simplify or rebalance
      ↓
Controlled retraining
```

Final objective:

```text
desired physical behavior
```

not:

```text
high total reward
```
