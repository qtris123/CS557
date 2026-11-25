# Locomotion Reward Design

Use this reference for quadrupeds, humanoids, bipeds, wheel-legged robots, and
other velocity- or trajectory-tracking locomotion tasks.

## Primary objective first

For command-conditioned locomotion, the minimum viable objective is usually:

```text
track commanded planar velocity
track commanded yaw rate
avoid true failure
```

Only add gait/style/energy terms after the policy can solve the basic command
tracking task.

Typical Isaac Lab built-ins include:

- `mdp.track_lin_vel_xy_exp`
- `mdp.track_ang_vel_z_exp`
- `mdp.lin_vel_z_l2`
- `mdp.ang_vel_xy_l2`
- `mdp.joint_torques_l2`
- `mdp.joint_acc_l2`
- `mdp.action_rate_l2`
- contact-related reward functions
- `mdp.is_terminated`

Check the version-matched Isaac Lab docs/source before copying exact API
signatures.

## Suggested conceptual hierarchy

```text
Reward
│
├── Task
│   ├── planar velocity tracking
│   └── yaw-rate tracking
│
├── Stability
│   ├── upright orientation
│   ├── vertical base velocity
│   └── roll/pitch angular velocity
│
├── Motion quality
│   ├── action rate
│   ├── joint acceleration
│   └── torque / power
│
├── Contact
│   ├── foot slip
│   ├── gait timing / air time (if justified)
│   └── undesired contacts
│
└── Failure
    └── true termination
```

## Minimal starting pattern

This is a structural example, not a universal set of weights:

```python
@configclass
class RewardsCfg:
    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    termination = RewTerm(func=mdp.is_terminated, weight=-5.0)
```

Treat every number above as a hypothesis to validate against actual raw term
magnitudes.

## Why tracking rewards often use exponential kernels

A bounded tracking reward such as:

```text
exp(-error^2 / sigma^2)
```

has useful properties:

- best value near exact tracking
- bounded contribution
- large errors stop dominating optimization
- `sigma` controls how forgiving the tracking objective is

If `sigma` is too small, the useful reward basin may be too narrow for early
exploration.

If `sigma` is too large, poor tracking may still receive strong reward.

Tune `sigma` based on command range and observed tracking errors, not by copying
a value from another robot.

## Standing-still failure mode

Symptom:

```text
robot is stable but ignores movement commands
```

Likely causes:

- energy penalty too strong
- torque penalty too strong
- acceleration penalty too strong
- contact penalty active during normal stepping
- tracking reward too weak or too narrow
- command distribution contains too many near-zero commands

Diagnosis:

1. compare zero-action episodic return with commanded-motion return
2. inspect weighted reward breakdown
3. temporarily remove the strongest movement-related penalty
4. verify command sampling
5. restore regularization with lower scale or curriculum

Do not solve this by blindly increasing all positive rewards.

## Slow-motion local optimum

Symptom:

```text
robot walks correctly but consistently below commanded speed
```

Interpretation:

The marginal reward for improved tracking may be smaller than the marginal cost
of moving faster.

Inspect:

- torque / power
- joint acceleration
- action rate
- contact force penalties
- slip penalty
- body stability penalties

Run an ablation that removes one suspected cost term while keeping PPO and all
other reward terms unchanged.

## Jerky motion

Before increasing `action_rate_l2`, verify:

- action scale
- policy output range
- control decimation
- actuator stiffness/damping
- observation delay/noise
- command discontinuities

If control semantics are correct and the policy is still unnecessarily jerky,
then add or strengthen smoothness regularization gradually.

## Upright/stability terms

Stability rewards can help, but they can also suppress legitimate dynamic
motion.

Potential conflict:

```text
high-speed locomotion requires body motion
vs.
strong orientation/angular-velocity penalties discourage body motion
```

For dynamic tasks, prefer the weakest stability regularization that removes
undesired behavior.

## Contact reward design

Contact terms are often the easiest way to accidentally dominate locomotion.

Always check:

- exact body-name selection
- sensor update period
- force threshold
- whether expected foot contact is excluded
- whether the penalty fires every step
- raw magnitude during healthy gait

If `undesired_contact` contributes more negative return than the tracking terms
contribute positive return during otherwise good locomotion, investigate the
sensor/threshold/weight before changing the optimizer.

## Foot air time and gait rewards

Do not add gait style rewards automatically.

Use them when:

- a specific gait is required
- the learned gait has a clear undesirable timing pattern
- hardware constraints justify a style preference

Risks:

- hopping becomes optimal
- exaggerated swing duration
- unnecessary foot lifting
- style reward competes with command tracking

Task performance should be learned before strict style.

## Zero-command behavior

Many reward conflicts appear only when commanded velocity is near zero.

Check whether:

- tracking correctly rewards standing
- gait/air-time reward still pushes stepping
- survival reward encourages useless motion
- orientation and contact terms behave sensibly

A term that makes sense at 1 m/s may be wrong at 0 m/s.

Consider command-conditioned gating for gait/style terms when appropriate.

## Curriculum pattern

Example strategy:

```text
Phase 1: learn movement
  tracking        strong
  stability       light
  action rate     very light
  energy          very light

Phase 2: improve quality
  tracking        unchanged
  stability       moderate
  action rate     moderate
  energy          light

Phase 3: robustness/style
  tracking        unchanged
  contact/style   task-dependent
  domain randomization stronger
```

Use Isaac Lab curriculum utilities when a reward term should change during
training.

## Recommended locomotion experiment sequence

```text
A. tracking + termination
B. + minimal stability
C. + action smoothness
D. + energy regularization
E. + contact/style only if needed
F. domain randomization and re-balance
```

At each step, keep the previous policy behavior and per-term reward breakdown as
the comparison baseline.

## Metrics worth logging

Beyond total reward, log:

- per-term episodic contribution
- commanded vs measured vx/vy/yaw rate
- velocity tracking RMSE
- fall rate
- episode length
- mean/max torque
- mechanical power or energy proxy
- action-rate norm
- undesired contact count
- foot slip proxy
- success across command-speed bins

Command-bin metrics are especially useful because a policy can have acceptable
mean reward while failing only at high-speed or turning commands.
