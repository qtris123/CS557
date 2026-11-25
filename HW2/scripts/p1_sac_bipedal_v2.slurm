#!/bin/bash
#SBATCH --job-name=p1_sac_bw2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p1_sac_bipedal_v2_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p1_sac_bipedal_v2_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2

/scratch/scholar/vo43/conda_envs/cs557rl/bin/python -u main.py --sac --env BipedalWalker-v3 \
    --lr 7e-4 --gamma 0.99 --tau 0.01 --alpha 0.2 \
    --batch_size 256 --replay_size 300000 --start_steps 10000 \
    --max_steps 500000 --save_interval 50000 --seed 42

echo "Job p1_sac_bw2 finished at $(date)"
