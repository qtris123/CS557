#!/bin/bash
# Part 2 Local Parallel Sweep Runner (aligned with submit_all_jobs.sh)
# Author: Tri Vo (vo43@purdue.edu)

set -e

HW_DIR="/home/vo43/CS557/HW2"
LOG_DIR="${HW_DIR}/logs"
CONDA_ENV="/scratch/scholar/vo43/conda_envs/cs557rl"

mkdir -p "${LOG_DIR}"
cd "${HW_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

echo "Starting Part 2A sweep experiments (200k steps, seed=42)..."

# 1. Clip ratio ablation (0.1, 0.3; 0.2 from Part 1B baseline)
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.1 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_clip01.out" 2>&1 &
PID1=$!
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.3 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_clip03.out" 2>&1 &
PID2=$!

# 2. Entropy coef ablation (0.0, 0.05; 0.01 from Part 1B baseline)
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.0 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_ent00.out" 2>&1 &
PID3=$!
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.05 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_ent05.out" 2>&1 &
PID4=$!

# 3. Learning rate ablation (1e-5, 3e-4, 1e-3; 5e-5 from Part 1B baseline)
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 1e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr1e5.out" 2>&1 &
PID5=$!
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr3e4.out" 2>&1 &
PID6=$!
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 1e-3 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr1e3.out" 2>&1 &
PID7=$!

# 4. Batch size ablation at lr=3e-4 (64, 128; 256 from lr3e4)
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 64 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_bs64.out" 2>&1 &
PID8=$!
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 128 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_bs128.out" 2>&1 &
PID9=$!

echo "Waiting for Part 2A sweep jobs to complete..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9

# 5. Part 2B Best Tuned PPO
echo "Running Part 2B Best Tuned PPO..."
${CONDA_ENV}/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 256 --rollout_steps 8192 --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --max_steps 2500000 --save_interval 200000 --seed 42 > "${LOG_DIR}/p2b_ppo_best.out" 2>&1

echo "All Part 2 jobs successfully completed!"
