#!/bin/bash
#SBATCH --job-name=p1_sac_mc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/vo43/CS557/HW2/logs/p1_sac_mc_%j.out
#SBATCH --error=/home/vo43/CS557/HW2/logs/p1_sac_mc_%j.err

echo "----------------------------------------------------"
echo "Job Name:    p1_sac_mc"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "Partition:   gpu"
echo "----------------------------------------------------"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd /home/vo43/CS557/HW2

/scratch/scholar/vo43/conda_envs/cs557rl/bin/python -u main.py --sac --env MountainCarContinuous-v0 --lr 3e-4 --gamma 0.99 --tau 0.01 --alpha 0.5 --batch_size 256 --replay_size 100000 --start_steps 5000 --ou_warmup --max_steps 200000 --save_interval 40000 --seed 42

echo "Job p1_sac_mc finished at $(date)"
