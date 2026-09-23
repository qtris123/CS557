#!/bin/bash
#SBATCH -A scholar
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C a40
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=openr1-sft
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

echo "============================================"
echo "  Open-R1 SFT Test Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Time:      $(date)"
echo "============================================"

# ══════════════════════════════════════════════════════════════
#  Background GPU Monitor — prints to .out and logs to CSV
# ══════════════════════════════════════════════════════════════
nvidia-smi

GPU_CSV="logs/gpu_monitor-${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_util_%,mem_used_MiB,mem_total_MiB,mem_util_%,temp_C,power_W" > "$GPU_CSV"

start_gpu_monitor() {
    while true; do
        # 1. Log machine-readable CSV every 5s
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw \
            --format=csv,noheader,nounits >> "$GPU_CSV" 2>/dev/null

        # 2. Print readable summary to stdout (captured in .out) every 15s
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

# Start GPU monitoring
start_gpu_monitor

# Fast test run: 30 steps with Qwen2.5-1.5B on Mixture-of-Thoughts (science)
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config science \
    --dataset_num_proc 8 \
    --eos_token '<|im_end|>' \
    --learning_rate 2.0e-5 \
    --max_steps 30 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --report_to none \
    --output_dir data/test-sft-1.5b

echo "Job finished at: $(date)"
