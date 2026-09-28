#!/bin/bash
#SBATCH -A scholar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C a40
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=openr1-grpo
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

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
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export VLLM_USE_V1=0

echo "============================================"
echo "  Open-R1 GRPO Test Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Time:      $(date)"
echo "============================================"

# Background GPU Monitor
GPU_CSV="logs/gpu_monitor-${SLURM_JOB_ID}.csv"
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

# Fast test run: 15 steps of GRPO on 1x A40 GPU
accelerate launch \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision bf16 \
    --num_processes 1 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --max_steps 15 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --report_to none \
    --push_to_hub false \
    --output_dir data/test-grpo-1.5b

echo "Job finished at: $(date)"
