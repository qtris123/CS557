#!/bin/bash
#SBATCH --job-name=ppo_batch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/ppo_batch_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/ppo_batch_%j.err

echo "----------------------------------------------------"
echo "Job Name:    ppo_batch (12 parallel PPO runs)"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "----------------------------------------------------"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

HW_DIR="/home/vo43/CS557/HW2"
LOG_DIR="${HW_DIR}/logs"
PY="/scratch/scholar/vo43/conda_envs/cs557rl/bin/python"

cd "${HW_DIR}"

${PY} -u main.py --ppo --env MountainCarContinuous-v0 --lr 3e-4 --gamma 0.999 --gae_lambda 0.9 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 64 --rollout_steps 2048 --norm_obs --use_sde --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p1_ppo_mc.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p1_ppo_bipedal.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.1 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_clip01.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.3 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_clip03.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.0 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_ent00.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.05 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_ent05.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 1e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr1e5.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr3e4.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 1e-3 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_lr1e3.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 64 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_bs64.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 128 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42 > "${LOG_DIR}/p2_ppo_bs128.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 --lr 3e-4 --clip_ratio 0.2 --entropy_coef 0.01 --batch_size 128 --gae_lambda 0.95 --gamma 0.99 --grad_clip 0.5 --anneal_lr --max_steps 700000 --save_interval 50000 --seed 42 > "${LOG_DIR}/p2b_ppo_best.out" 2>&1 &

wait
echo "All 12 PPO runs completed at $(date)"
