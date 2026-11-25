#!/bin/bash
#SBATCH --job-name=p2b_ppo_v5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:15:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p2b_ppo_v5_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p2b_ppo_v5_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2
LOG_DIR="/home/vo43/CS557/HW2/logs"
PY="/scratch/scholar/vo43/conda_envs/cs557rl/bin/python"

${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 3e-4 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 64 \
    --gae_lambda 0.95 --gamma 0.999 --grad_clip 0.5 --anneal_lr --td_lambda_returns \
    --use_sde --sde_sample_freq 4 --log_std_init -1.0 \
    --max_steps 2000000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v5_sde_lr3e4.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 5e-4 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns \
    --use_sde --sde_sample_freq 4 --log_std_init -1.0 \
    --max_steps 2000000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v5_sde_lr5e4.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns --use_tanh \
    --max_steps 2000000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v5_tanh_lr1e3_2m.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 5e-4 --clip_ratio 0.2 --entropy_coef 0.001 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --td_lambda_returns \
    --max_steps 2000000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v5_relu_lr5e4_ent001.out" 2>&1 &
wait
echo "p2b_ppo_v5 finished at $(date)"
