#!/bin/bash
#SBATCH --job-name=p1_reinforce_bipedal
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p1_reinforce_bipedal_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p1_reinforce_bipedal_%j.err

echo "----------------------------------------------------"
echo "Job Name:    p1_reinforce_bipedal"
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

/scratch/scholar/vo43/conda_envs/cs557rl/bin/python -u main.py --reinforce --env BipedalWalker-v3 --lr 5e-5 --max_steps 200000 --save_interval 50000 --seed 42

echo "Job p1_reinforce_bipedal finished at $(date)"
