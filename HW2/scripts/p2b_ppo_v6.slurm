#!/bin/bash
#SBATCH --job-name=p2b_ppo_v6
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p2b_ppo_v6_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p2b_ppo_v6_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2
LOG_DIR="/home/vo43/CS557/HW2/logs"
PY="/scratch/scholar/vo43/conda_envs/cs557rl/bin/python"

${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 7e-4 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr \
    --max_steps 1200000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v6_lr7e4_g995.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.15 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr \
    --max_steps 1200000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v6_lr1e3_clip15.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr \
    --max_steps 1200000 --save_interval 100000 --seed 1 > "${LOG_DIR}/p2b_v6_lr1e3_s1.out" 2>&1 &
sleep 1
${PY} -u main.py --ppo --env BipedalWalker-v3 \
    --lr 7e-4 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 128 \
    --gae_lambda 0.95 --gamma 0.999 --grad_clip 0.5 --anneal_lr \
    --max_steps 1200000 --save_interval 100000 --seed 42 > "${LOG_DIR}/p2b_v6_lr7e4_g999.out" 2>&1 &
wait
echo "p2b_ppo_v6 finished at $(date)"
