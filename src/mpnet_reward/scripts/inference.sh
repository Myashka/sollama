#!/bin/bash

python inference_script.py \
  --use_title True \
  --t_col "Title" \
  --q_col "Question" \
  --a_col "Generated Answer" \
  --postfix "_gen" \
  --max_length_q 512 \
  --max_length_a 256 \
  --batch_size 64 \
  --file_path "/home/st-gorbatovski/sollama/data/processed/tanh_score/python_basics/hand_check/test-sft-LoRA_llama-7b-max_prompt_length_512-t_09-basics-paraphrased-3_attempts_4.csv" \
  --model_name "/home/st-gorbatovski/sollama/src/mpnet_reward/artifacts/experiments/train-mpnet-so_par-lr_2e5-not_norm-so_margin_5-dot_prod_sim-fr_1_10_11/checkpoint-14345" \
  --torch_dtype null \
  --device_map "cuda:2" \
  --padding_side "right" \
  --seed 42
