#!/usr/bin/env bash
set -euo pipefail

# =========================
# Open-source friendly SFT launcher (LLaMA-Factory)
# =========================
# Usage:
#   bash scripts/train_llamafactory_qwen3vl_lora.sh \
#     --llamafactory_dir /path/to/LLaMA-Factory \
#     --model Qwen/Qwen3-VL-8B-Instruct \
#     --dataset_name tumor_vqa_all_cot_clean_final_first_shuffle1 \
#     --dataset_dir /path/to/llamafactory/data \
#     --output_dir ./saves/qwen3vl_lora_cot_1epoch
#
# Notes:
# - This script does NOT bundle LLaMA-Factory.
# - Users should clone LLaMA-Factory themselves and prepare datasets in its expected format.
# - All private absolute paths are removed.

# -------- defaults (override via CLI or env) --------
CUDA_VISIBLE_DEVICES_DEFAULT="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT_DEFAULT="${MASTER_PORT:-29515}"
NPROC_PER_NODE_DEFAULT="${NPROC_PER_NODE:-4}"

# NCCL settings (optional; keep as conservative defaults for multi-GPU)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# Disable WandB by default (can override)
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

# -------- parse args --------
LLAMAFACTORY_DIR=""
MODEL_NAME_OR_PATH=""
DATASET_NAME=""
DATASET_DIR=""
OUTPUT_DIR=""

PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
TEMPLATE="${TEMPLATE:-qwen2_vl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llamafactory_dir) LLAMAFACTORY_DIR="$2"; shift 2;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2;;
    --dataset_name) DATASET_NAME="$2"; shift 2;;
    --dataset_dir) DATASET_DIR="$2"; shift 2;;
    --output_dir) OUTPUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$LLAMAFACTORY_DIR" || -z "$MODEL_NAME_OR_PATH" || -z "$DATASET_NAME" || -z "$DATASET_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required args." >&2
  echo "Required: --llamafactory_dir --model --dataset_name --dataset_dir --output_dir" >&2
  exit 1
fi

# -------- run --------
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_DEFAULT"

echo "🚀 Launching LoRA SFT (LLaMA-Factory)"
echo "  LLaMA-Factory:  $LLAMAFACTORY_DIR"
echo "  Model:          $MODEL_NAME_OR_PATH"
echo "  Dataset:        $DATASET_NAME"
echo "  Dataset dir:    $DATASET_DIR"
echo "  Output dir:     $OUTPUT_DIR"
echo "  GPUs:           $CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE_DEFAULT, port=$MASTER_PORT_DEFAULT)"

cd "$LLAMAFACTORY_DIR"

python -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE_DEFAULT" \
  --master_port="$MASTER_PORT_DEFAULT" \
  src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset "$DATASET_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --template "$TEMPLATE" \
    --finetuning_type lora \
    --lora_target "$LORA_TARGET" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr_scheduler_type cosine \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --plot_loss \
    --bf16 \
    --ddp_find_unused_parameters False \
    --warmup_ratio "$WARMUP_RATIO" \
    --trust_remote_code True \
    --report_to none
