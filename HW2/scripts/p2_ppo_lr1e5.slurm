#!/bin/bash
#SBATCH --job-name=p2_ppo_lr1e5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p2_ppo_lr1e5_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p2_ppo_lr1e5_%j.err

echo "----------------------------------------------------"
echo "Job Name:    p2_ppo_lr1e5"
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

/scratch/scholar/vo43/conda_envs/cs557rl/bin/python -u main.py --ppo --env BipedalWalker-v3 --lr 1e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42

echo "Job p2_ppo_lr1e5 finished at $(date)"
