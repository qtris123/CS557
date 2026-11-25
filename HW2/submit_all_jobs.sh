#!/bin/bash
# Master Slurm submission script for CS 557 HW2 Experiments
# Author: Tri Vo (vo43@purdue.edu)
# Purdue University - Scholar Cluster

set -e

HW_DIR="/home/vo43/CS557/HW2"
LOG_DIR="${HW_DIR}/logs"
SCRIPTS_DIR="${HW_DIR}/scripts"
CONDA_ENV="/scratch/scholar/vo43/conda_envs/cs557rl"

mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPTS_DIR}"
mkdir -p "${HW_DIR}/checkpoints"
mkdir -p "${HW_DIR}/runs"

echo "=== CS 557 Homework 2: Submitting Clean Cluster Jobs ==="
echo "Working directory: ${HW_DIR}"
echo "Conda environment: ${CONDA_ENV}"
echo "Logs directory: ${LOG_DIR}"

submit_job() {
    local job_name="$1"
    local command_args="$2"
    local partition="${3:-cpu}"
    local time_limit="${4:-01:00:00}"
    local script_file="${SCRIPTS_DIR}/${job_name}.slurm"

    local gpu_line=""
    if [ "${partition}" = "gpu" ]; then
        gpu_line="#SBATCH --gres=gpu:1"
    fi

    cat << EOF > "${script_file}"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
${gpu_line}
#SBATCH --mem=8G
#SBATCH --time=${time_limit}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err

echo "----------------------------------------------------"
echo "Job Name:    ${job_name}"
echo "Node:        \$(hostname)"
echo "Date:        \$(date)"
echo "Partition:   ${partition}"
echo "----------------------------------------------------"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1

cd ${HW_DIR}

${CONDA_ENV}/bin/python -u main.py ${command_args}

echo "Job ${job_name} finished at \$(date)"
EOF

    local job_id
    job_id=$(sbatch "${script_file}" | awk '{print $4}')
    echo "Submitted [${job_name}] -> Job ID: ${job_id}"
}

echo ""
echo "--- Part 1: Baseline & Solved Agents ---"
# 1. REINFORCE MountainCarContinuous-v0
submit_job "p1_reinforce_mc" "--reinforce --env MountainCarContinuous-v0 --lr 5e-5 --max_steps 200000 --save_interval 50000 --seed 42" "cpu" "00:30:00"

# 2. REINFORCE BipedalWalker-v3
submit_job "p1_reinforce_bipedal" "--reinforce --env BipedalWalker-v3 --lr 5e-5 --max_steps 200000 --save_interval 50000 --seed 42" "cpu" "00:30:00"

# 3. PPO MountainCarContinuous-v0 (obs normalization + gSDE + gamma=0.999)
submit_job "p1_ppo_mc" "--ppo --env MountainCarContinuous-v0 --lr 3e-4 --gamma 0.999 --gae_lambda 0.9 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 64 --rollout_steps 2048 --norm_obs --use_sde --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

# 4. PPO BipedalWalker-v3 (Part 1B / Part 2A Starter Baseline)
submit_job "p1_ppo_bipedal" "--ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

# 5. SAC MountainCarContinuous-v0 (OU warmup + fixed truncation & alpha detach)
submit_job "p1_sac_mc" "--sac --env MountainCarContinuous-v0 --lr 3e-4 --gamma 0.99 --tau 0.01 --alpha 0.5 --batch_size 256 --replay_size 100000 --start_steps 5000 --ou_warmup --max_steps 200000 --save_interval 40000 --seed 42" "gpu" "00:45:00"

# 6. SAC BipedalWalker-v3 (fixed truncation & alpha detach, 500k steps)
submit_job "p1_sac_bipedal" "--sac --env BipedalWalker-v3 --lr 7e-4 --gamma 0.99 --tau 0.01 --alpha 0.2 --batch_size 256 --replay_size 300000 --start_steps 10000 --max_steps 500000 --save_interval 50000 --seed 42" "gpu" "02:00:00"

echo ""
echo "--- Part 2A: Hyperparameter Sweep for PPO on BipedalWalker-v3 ---"
# 1. Clip ratio ablation (0.1, 0.2 [baseline], 0.3)
submit_job "p2_ppo_clip01" "--ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.1 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"
submit_job "p2_ppo_clip03" "--ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.3 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

# 2. Entropy coef ablation (0.0, 0.01 [baseline], 0.05)
submit_job "p2_ppo_ent00" "--ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.0 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"
submit_job "p2_ppo_ent05" "--ppo --env BipedalWalker-v3 --lr 5e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.05 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

# 3. Learning rate ablation (1e-5, 5e-5 [baseline], 3e-4, 1e-3)
submit_job "p2_ppo_lr1e5" "--ppo --env BipedalWalker-v3 --lr 1e-5 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"
submit_job "p2_ppo_lr3e4" "--ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"
submit_job "p2_ppo_lr1e3" "--ppo --env BipedalWalker-v3 --lr 1e-3 --batch_size 256 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

# 4. Batch size ablation at lr=3e-4 (64, 128, 256 [from lr3e4])
submit_job "p2_ppo_bs64" "--ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 64 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"
submit_job "p2_ppo_bs128" "--ppo --env BipedalWalker-v3 --lr 3e-4 --batch_size 128 --clip_ratio 0.2 --entropy_coef 0.01 --max_steps 200000 --save_interval 40000 --seed 42" "cpu" "00:30:00"

echo ""
echo "--- Part 2B: Best Tuned Hyperparameter List (Full Budget to Solve >= 300) ---"
submit_job "p2b_ppo_best" "--ppo --env BipedalWalker-v3 --lr 1e-3 --clip_ratio 0.2 --entropy_coef 0.0 --batch_size 256 --rollout_steps 8192 --gae_lambda 0.95 --gamma 0.995 --grad_clip 0.5 --anneal_lr --max_steps 2500000 --save_interval 200000 --seed 42" "cpu" "01:00:00"

echo ""
echo "All 16 jobs successfully submitted to Slurm!"
