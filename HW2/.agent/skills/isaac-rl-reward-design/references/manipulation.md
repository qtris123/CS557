# Manipulation Reward Design

Use this reference for reaching, grasping, pick-and-place, insertion, transport,
object reorientation, dexterous manipulation, and other staged manipulation
tasks.

## Manipulation rewards are usually staged

A manipulation task often contains a causal sequence:

```text
reach
  ↓
contact
  ↓
grasp
  ↓
lift
  ↓
transport
  ↓
place / insert
  ↓
success
```

Do not automatically expose every positive shaping term at full strength from
the first timestep.

When later-stage rewards can be earned without completing earlier stages, reward
hacking becomes likely.

## Start from the task success definition

Write an explicit success predicate before designing shaping rewards.

Example pick-and-place success:

```text
object center is within target tolerance
AND object height/orientation is valid
AND object is stable for N steps
```

Success should represent the desired final state, not an intermediate proxy such
as gripper-object distance.

## Minimum viable decomposition

A generic task can be written conceptually as:

```text
reward =
    reach_progress
  + valid_grasp
  + lift_progress
  + target_progress
  + success_bonus
  - unsafe_or_wasteful_motion
```

The exact terms should be task-dependent.

## Reach reward

Typical form:

```text
exp(-distance^2 / sigma^2)
```

Potential exploit:

```text
grender stays near object forever without grasping
```

Mitigations:

- cap the reach contribution
- reduce or disable reach reward after grasp
- reward progress rather than absolute proximity when appropriate
- make later-stage reward clearly more valuable

Do not make proximity the dominant episodic reward when the real goal is object
manipulation.

## Contact reward

Contact is not the same as a valid grasp.

A simple contact bonus can be exploited by:

- poking
- tapping repeatedly
- trapping the object against the table
- making contact with the wrong link

Prefer a task-specific grasp predicate using evidence such as:

- correct finger/contact bodies
- opposing contacts
- gripper closure state
- object motion coupled with the hand
- object supported after lift

The exact implementation depends on the robot and sensor setup.

## Grasp reward

A valid-grasp reward should be difficult to collect accidentally.

Potential implementation pattern:

```text
grasp_valid =
    left_finger_contact
    AND right_finger_contact
    AND gripper_width_within_range
```

This is only a conceptual example.

For suction, magnetic, dexterous, or underactuated hands, use a validity test
that matches the real grasp mechanism.

## Lift reward

Reward object height only after a valid grasp when possible.

Bad pattern:

```text
reward += object_height
```

Possible exploit:

```text
strike / throw the object upward
```

Safer conceptual pattern:

```text
if grasp_valid:
    reward += lift_progress
```

Also check object vertical velocity: a ballistic object may have height but not
be under control.

## Target transport reward

Activate target-distance shaping after the object has been lifted or otherwise
secured.

Potential conflict:

If target reward is active before grasp, the robot may push or knock the object
toward the target instead of performing the intended grasp-and-carry behavior.

Whether pushing is an exploit depends on the real task specification.

## Placement / insertion

For precise placement or insertion, decompose errors only as much as necessary:

- position error
- orientation error
- insertion depth
- contact/force safety
- stability after release

Avoid heavily weighting all errors simultaneously if early exploration cannot
enter the useful region.

Consider curricula on:

- target tolerance
- object initial pose range
- target distance
- orientation range
- insertion depth

## Success bonus

A success bonus is useful when a dense shaping path already gets the policy near
the true goal.

Risks:

- repeated triggering every timestep
- oscillating in/out of success threshold
- continuing the episode only to farm success reward

Possible mitigations:

- one-time success event
- terminate on success when appropriate
- require success persistence
- use a boolean latch

## Progress vs absolute-state rewards

Absolute-state reward:

```text
reward = -distance_to_target
```

Progress reward:

```text
reward = previous_distance - current_distance
```

Progress rewards directly pay for improvement, but can introduce issues around
reset boundaries, noise, oscillation, and discounting.

Choose based on task dynamics; do not assume progress reward is always superior.

## Gating pattern

A useful conceptual structure is:

```python
reward = 0.0

reward += reach_reward

if grasp_valid:
    reward += grasp_bonus
    reward += lift_reward

if lifted:
    reward += target_reward

if inside_target:
    reward += placement_reward

if success:
    reward += success_bonus
```

This is a design pattern, not copy-paste Isaac Lab code.

Ask whether each gate is:

- observable from simulation state
- robust to noise
- hard to exploit
- compatible with sim-to-real assumptions

## Avoid over-gating

Too many discrete stages can create discontinuities and block learning.

Warning signs:

- policy reaches a gate boundary but cannot cross it
- a later-stage reward is never observed
- success rate stays exactly zero while early shaping saturates

Possible fixes:

- overlap adjacent stages
- keep weak continuous shaping across the boundary
- widen tolerances early
- use curriculum

## Action regularization

Manipulation often needs less aggressive motion regularization early than people
expect.

If action-rate or torque penalties are too strong, the arm may approach very
slowly or refuse to close the gripper.

First learn reliable task progression; then tune motion quality.

For hardware deployment, later add constraints motivated by:

- joint velocity/acceleration limits
- torque/current limits
- collision safety
- contact-force limits
- gripper wear

## Reward hacking checklist

For each term ask:

### Reach
Can the robot hover near the object indefinitely?

### Contact
Can it tap repeatedly?

### Grasp
Can it trigger the predicate without supporting the object?

### Lift
Can it throw or bounce the object?

### Target
Can it push the object instead of carrying it?

### Placement
Can it enter the tolerance at high speed and immediately leave?

### Success
Can it retrigger the success bonus?

### Termination
Can it intentionally terminate to avoid accumulating penalties?

## Curriculum examples

Useful curricula include:

```text
easy initial object poses -> wider pose distribution
near target               -> farther target
large placement tolerance -> tight tolerance
fixed orientation         -> randomized orientation
weak energy penalty       -> stronger hardware-aware penalty
```

Do not make the reward itself carry every difficulty dimension. Curriculum can
modify task distribution as well as reward weights.

## Recommended experiment sequence

For pick-and-place:

```text
A. reach only: verify exploration and controller semantics
B. reach + valid grasp
C. + gated lift
D. + gated target transport
E. + true success
F. add smoothness/safety regularization
G. broaden randomization
```

This does not mean training seven entirely separate production policies. It is a
debugging sequence for discovering where learning fails.

## Metrics worth logging

- reach distance
- valid-grasp rate
- lift success rate
- max controlled object height
- target distance conditioned on valid grasp
- placement success rate
- success persistence
- object drop rate
- collision count
- action-rate norm
- torque/power proxy
- per-stage occupancy time
- per-term episodic reward

Stage-conditioned metrics are often more informative than total reward because
they show exactly where the manipulation chain breaks.
