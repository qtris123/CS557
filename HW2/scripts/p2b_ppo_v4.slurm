#!/bin/bash
#SBATCH --job-name=p2b_ppo_v4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p2b_ppo_v4_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p2b_ppo_v4_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2
LOG_DIR="/home/vo43/CS557/HW2/logs"
PY="/scratch/scholar/vo43/conda_envs/cs557rl/bin/python"

${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns \
    --max_steps 1000000 --save_interval 50000 --seed 42 > "${LOG_DIR}/p2b_v4_tdlam_s42.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns --use_tanh \
    --max_steps 1000000 --save_interval 50000 --seed 42 > "${LOG_DIR}/p2b_v4_tdlam_tanh_s42.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 5e-4 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 64 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns --use_tanh \
    --max_steps 1000000 --save_interval 50000 --seed 42 > "${LOG_DIR}/p2b_v4_lr5e4_bs64_tanh.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns \
    --max_steps 1000000 --save_interval 50000 --seed 0 > "${LOG_DIR}/p2b_v4_tdlam_s0.out" 2>&1 &
wait
echo "p2b_ppo_v4 finished at $(date)"
