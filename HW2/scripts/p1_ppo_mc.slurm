#!/bin/bash
#SBATCH --job-name=p1_ppo_mc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p1_ppo_mc_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p1_ppo_mc_%j.err

echo "----------------------------------------------------"
echo "Job Name:    p1_ppo_mc"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Partition:   cpu"
echo "----------------------------------------------------"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2

/scratch/scholar/vo43/conda_envs/cs557rl/bin/python -u main.py --ppo --env MountainCarContinuous-v0 --lr 3e-4 --gamma 0.999 --gae_lambda 0.9 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 64 --rollout_steps 2048 --norm_obs --use_sde --max_steps 200000 --save_interval 40000 --seed 42

echo "Job p1_ppo_mc finished at $(date)"
