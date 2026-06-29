#!/usr/bin/env bash
set -u

ROOT="/Users/sxc/Desktop/去年的SRTP"
CODE_DIR="$ROOT/SRTP/HVFormer-main/HVFormer-main"
PY="$ROOT/paper_submission/.venv-mps/bin/python"
OUT_DIR="$ROOT/paper_submission/results/strong_search_20epoch"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR/raw"

cd "$CODE_DIR" || exit 1

run_exp() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
  printf '\n===== %s =====\n' "$name" | tee -a "$LOG_DIR/queue.log"
  printf 'Command: %s\n' "$*" | tee -a "$LOG_DIR/queue.log"
  "$PY" scripts/train_ner_baselines.py "$@" 2>&1 | tee "$log_file"
}

run_exp bert_linear_caption_seed1_20 \
  --dataset_name caption_ner \
  --model_type bert_linear \
  --bert_name bert-base-uncased \
  --num_epochs 20 \
  --batch_size 16 \
  --max_seq 80 \
  --device mps \
  --seed 1 \
  --lr 3e-5 \
  --warmup_ratio 0.06 \
  --log_steps 40 \
  --tune_decode_o_bias

run_exp bert_linear_caption_paired_context_seed1_20 \
  --dataset_name caption_ner \
  --model_type bert_linear \
  --bert_name bert-base-uncased \
  --num_epochs 20 \
  --batch_size 16 \
  --max_seq 128 \
  --device mps \
  --seed 1 \
  --lr 3e-5 \
  --warmup_ratio 0.06 \
  --log_steps 40 \
  --paired_context auto \
  --context_max_tokens 48 \
  --tune_decode_o_bias

run_exp bert_boundary_caption_paired_context_seed1_20 \
  --dataset_name caption_ner \
  --model_type bert_boundary \
  --bert_name bert-base-uncased \
  --num_epochs 20 \
  --batch_size 16 \
  --max_seq 128 \
  --device mps \
  --seed 1 \
  --lr 3e-5 \
  --warmup_ratio 0.06 \
  --log_steps 40 \
  --boundary_loss_weight 0.2 \
  --paired_context auto \
  --context_max_tokens 48 \
  --tune_decode_o_bias

run_exp roberta_linear_caption_seed1_20 \
  --dataset_name caption_ner \
  --model_type bert_linear \
  --bert_name roberta-base \
  --num_epochs 20 \
  --batch_size 16 \
  --max_seq 80 \
  --device mps \
  --seed 1 \
  --lr 2e-5 \
  --warmup_ratio 0.06 \
  --log_steps 40 \
  --tune_decode_o_bias

run_exp roberta_linear_caption_paired_context_seed1_20 \
  --dataset_name caption_ner \
  --model_type bert_linear \
  --bert_name roberta-base \
  --num_epochs 20 \
  --batch_size 16 \
  --max_seq 128 \
  --device mps \
  --seed 1 \
  --lr 2e-5 \
  --warmup_ratio 0.06 \
  --log_steps 40 \
  --paired_context auto \
  --context_max_tokens 48 \
  --tune_decode_o_bias

find "$CODE_DIR/reports/baseline_results" -name '*.json' -mmin -360 -print0 \
  | xargs -0 -I{} cp {} "$OUT_DIR/raw/"
