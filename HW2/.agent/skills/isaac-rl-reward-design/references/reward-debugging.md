# Reward Debugging Playbook

Use this reference when training is unstable, behavior is strange, reward is
rising without better behavior, or a `RewardsCfg` needs review.

## First principle

Do not debug `total_reward` in isolation.

A rising total reward can mean:

- genuine task improvement
- exploiting a shaping term
- avoiding a penalty by doing less
- longer episodes without better task performance
- a curriculum changed the reward scale

Always connect reward curves to physical behavior and task metrics.

## Debugging order

Use this order before changing PPO aggressively:

```text
1. Environment correctness
2. Observation correctness
3. Command/task distribution
4. Action scaling and control semantics
5. Reward implementation
6. Termination implementation
7. Reward scale / interactions
8. Curriculum
9. PPO hyperparameters
```

## Environment correctness

Check:

- simulation timestep
- control decimation
- frame conventions
- unit conventions
- asset body/joint names
- sensor update rates
- reset state
- randomization ranges

A reward cannot repair an invalid state signal.

## Observation correctness

Check:

- tensor ordering and dimensions
- normalization/scaling
- command visibility to the policy
- sign conventions
- body/world frame transformations
- stale sensor data
- clipping

If the policy does not observe information required to optimize a reward, the
reward may become impossible or exploitable.

## Action semantics

Check:

- position vs velocity vs effort action
- action scale
- action offset/default pose
- policy output clipping
- actuator limits
- control frequency

Large action scaling can make a smooth policy look jerky.

Tiny action scaling can make the robot appear unwilling to move even when reward
is correct.

## Reward implementation audit

For every custom reward:

1. inspect sign
2. inspect reduction (`sum`, `mean`, norm dimension)
3. inspect units
4. inspect body/joint selection
5. inspect thresholds
6. inspect clipping
7. inspect timestep dependence
8. inspect zero-command behavior
9. inspect reset behavior

Common bug:

```text
sum over 12 joints
```

is compared directly with:

```text
mean over 2 velocity axes
```

and equal-looking weights are assumed to have equal influence.

They do not.

## Reward scale audit

For each term collect:

```text
raw mean
raw std
raw p95
raw max abs
weight
weighted mean
weighted episodic sum
```

If using Isaac Lab ManagerBased environments, remember that the reward manager
applies the environment `dt` to term weights.

Do not double-apply `dt` in custom terms unless the custom quantity intentionally
represents something different.

## Positive/negative budget

Separate episodic positive and negative contributions.

Example:

```text
Positive:
  track_lin_vel       +18
  track_ang_vel        +7

Negative:
  orientation          -2
  action_rate          -3
  joint_acc            -1
  torque               -0.2
  undesired_contact   -40
```

The first experiment should target `undesired_contact`.

Do not react by increasing all positive weights.

## Dominant term checklist

When one auxiliary term dominates, inspect:

- raw scale
- whether it is summed over many bodies/joints
- threshold too low
- sensor firing continuously
- sign error
- missing gate
- wrong body regex
- unit mismatch
- task state where it should be inactive

## Near-zero term checklist

A term can also be useless.

If weighted contribution stays near zero, ask:

- is the raw term almost always zero?
- is weight too small?
- is a condition never reached?
- is body/joint selection empty or wrong?
- is the term redundant with another term?

Do not keep dead terms for decoration.

## Correlation with task metrics

Reward curves should be compared with task metrics.

Examples:

Locomotion:

- velocity RMSE
- fall rate
- high-speed success
- turning success

Manipulation:

- grasp rate
- lift rate
- placement success
- object drop rate

If reward rises but task metrics do not, investigate exploitation or scale
changes.

## Behavior-to-reward diagnosis

### Robot does not move

Inspect:

- tracking reward magnitude
- torque penalty
- energy penalty
- action penalty
- action-rate penalty
- joint acceleration penalty
- normal contact penalties
- command sampling

Hypothesis:

```text
movement cost > task value
```

Experiment:

Remove the single largest movement-related penalty and retrain with everything
else unchanged.

### Robot moves too slowly

Inspect whether faster motion increases:

- torque
- power
- acceleration
- contact force
- slip
- orientation penalties

Plot reward contribution against commanded speed bins if possible.

### Robot shakes

Check control semantics before reward:

- action scale
- control frequency
- stiffness/damping
- delayed/noisy observations

Then inspect:

- action-rate term
- acceleration term

### Robot falls when commands get hard

Bin metrics by command magnitude.

The global mean can hide a failure boundary.

Inspect:

- high-command tracking reward basin
- curriculum difficulty
- stability regularization
- actuator saturation
- termination rate

### Robot gets high reward with bizarre behavior

Assume reward hacking until disproven.

Find the reward term whose contribution increases most during the bizarre
behavior.

### Manipulator hovers near object

Likely cause:

- reach/proximity reward too profitable
- later-stage reward inaccessible or weak

Try:

- reduce/cap reach reward
- disable it after grasp
- improve grasp gradient
- use staged activation

### Manipulator throws object

Likely cause:

- height/lift reward does not require valid grasp

Gate lift reward on grasp/control state.

## Threshold discontinuities

Rewards based on conditions such as:

```text
if distance < 0.05: +1
```

can create:

- oscillation around the boundary
- sparse gradients
- sensitivity to noise

Options:

- add smooth shaping around the threshold
- add hysteresis
- use a one-time latch
- require persistence

Use discrete bonuses for meaningful events, not as the only learning signal for
difficult exploration.

## Termination audit

For every termination condition ask:

- is it a true failure?
- is it a timeout?
- can policy intentionally trigger it?
- does early termination avoid future negative reward?
- is success also a termination?

If failing early avoids a long stream of penalties, the policy may learn to
terminate intentionally.

## Zero-action baseline

Run a zero-action or default-pose policy.

Record:

- total episodic return
- every reward term
- episode length
- termination reason

This baseline answers:

```text
How expensive is attempting the task compared with doing nothing?
```

## Random-action baseline

Use safe random actions if the environment permits.

Look for:

- positive contact farming
- accidental success triggers
- excessive survival reward
- high random shaping reward

The purpose is not performance; it is exploit discovery.

## Scripted successful trajectory

If possible, evaluate a heuristic/controller/demo trajectory.

Ask whether a trajectory known to be good receives higher return than:

- zero action
- random action
- known bad behavior

If not, fix reward alignment before long RL runs.

## Ablation protocol

Good ablation:

```text
Baseline:
  original reward

Experiment 1:
  remove one suspected term

Experiment 2:
  restore it at 0.2x weight

Experiment 3:
  restore via curriculum
```

Bad ablation:

```text
change 8 reward weights
change PPO LR
change network
change action scale
increase randomization
```

You cannot infer causality from the second setup.

## Reward-weight sweep

Only sweep a weight after identifying the relevant term.

Prefer a coarse multiplicative sweep:

```text
0
0.1x
0.3x
1x
3x
```

rather than many tiny additive changes.

Compare physical task metrics, not only final reward.

## Curriculum diagnosis

If behavior changes after a curriculum milestone, remember the reward scale may
also change.

Plot or record:

- curriculum phase
- current reward weights
- command/randomization difficulty

A discontinuity in total reward at the curriculum transition is not necessarily
policy collapse.

## PPO should be later

After environment and reward checks pass, inspect PPO issues such as:

- learning rate
- clip range
- entropy
- value loss
- advantage normalization
- horizon
- batch size
- observation normalization

Reward problems and optimizer problems can look similar, but reward alignment
must be established first.

## Review template

Use this table when reviewing `RewardsCfg`:

| Term | Category | Raw scale | Weighted episodic scale | Risk | Action |
|---|---|---:|---:|---|---|

Then summarize:

### Overall diagnosis

List the 1–3 highest-confidence problems.

### Remove

Terms that are redundant, dead, or actively harmful.

### Modify

Terms with wrong scale, gating, threshold, body selection, or function shape.

### Missing

Only add new terms when observed behavior justifies them.

### Controlled experiments

For each experiment provide:

- hypothesis
- exact change
- expected behavior
- success metric

## Minimum evidence standard

A reward-change recommendation should ideally be supported by at least one of:

- per-term reward magnitude
- rollout behavior
- task metric
- code inspection showing a conflict
- controlled ablation result

If evidence is missing, label the recommendation as a hypothesis rather than a
fact.
