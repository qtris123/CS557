#!/bin/bash
#SBATCH -A scholar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -C a40
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=openr1-profile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs
mkdir -p /scratch/scholar/vo43/open-r1/profiles
mkdir -p /scratch/scholar/vo43/open-r1/data

module load cuda/12.1.0
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate openr1

cd /home/vo43/CS557/open-r1
export PYTHONPATH="src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/scratch/scholar/vo43/.triton
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
# Note: For 2 GPUs, Accelerate manages RANK and WORLD_SIZE dynamically
export VLLM_USE_V1=0

# Ensure profiles directory is symlinked to /scratch
if [ ! -L "profiles" ]; then
    ln -s /scratch/scholar/vo43/open-r1/profiles profiles 2>/dev/null || true
fi

PROFILE_OUTPUT="/scratch/scholar/vo43/open-r1/profiles/grpo_2gpu_${SLURM_JOB_ID}"

echo "============================================"
echo "  Open-R1 GRPO Nsight Profiling Job"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $SLURM_NODELIST"
echo "  GPU(s):       $CUDA_VISIBLE_DEVICES"
echo "  Profile File: ${PROFILE_OUTPUT}.nsys-rep"
echo "  Time:         $(date)"
echo "============================================"

nvidia-smi

# Background GPU Monitor (saves CSV to scratch)
GPU_CSV="/scratch/scholar/vo43/open-r1/profiles/gpu_monitor-${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_util_%,mem_used_MiB,mem_total_MiB,mem_util_%,temp_C,power_W" > "$GPU_CSV"

start_gpu_monitor() {
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw \
            --format=csv,noheader,nounits >> "$GPU_CSV" 2>/dev/null

        echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] GPU Status ==="
        nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,nounits
        echo "───────────────────────────────────────────────"

        sleep 15
    done &
    GPU_MONITOR_PID=$!
    echo "📊 GPU monitor started (PID=$GPU_MONITOR_PID)"
}

cleanup() {
    if [ -n "$GPU_MONITOR_PID" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null
        wait "$GPU_MONITOR_PID" 2>/dev/null
        echo "📊 GPU monitor stopped."
    fi
}
trap cleanup EXIT INT TERM

start_gpu_monitor

# Launch GRPO training wrapped in NVIDIA Nsight Systems (nsys):
#   -t cuda,nvtx,osrt : Trace CUDA API/kernels, NVTX ranges, and OS runtime
#   -s none           : Disable CPU backtrace sampling (avoids massive files and overhead)
#   -y 70             : Delay profiling by 70s to skip model loading and vLLM KV-cache allocation
#   --stats=true      : Print formatted summary statistics tables directly into SLURM log
#   -o <path>         : Output .nsys-rep strictly to /scratch
nsys profile \
    -t cuda,nvtx,osrt \
    -s none \
    -y 70 \
    --stats=true \
    --force-overwrite=true \
    -o "$PROFILE_OUTPUT" \
    accelerate launch \
        --num_machines 1 \
        --dynamo_backend no \
        --mixed_precision bf16 \
        --num_processes 2 \
        src/open_r1/grpo.py \
        --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.3 \
        --max_steps 6 \
        --num_generations 4 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --report_to none \
        --push_to_hub false \
        --overwrite_output_dir true \
        --output_dir data/profile-grpo-2gpu

echo "Profiling complete at: $(date)"
echo "Profile saved to: ${PROFILE_OUTPUT}.nsys-rep"
