python bert_eval.py --file_path /home/st-gorbatovski/sollama/src/sft/artifacts/tests/python_basics/test-rlhf-LoRA_llama-7b-t_09-basics-mpnet_dot_prod_so_par-cos_sim-length_penalty-step_261-mean_reward_0.48-3_attempts.csv --device cuda:2
python bert_eval.py --file_path /home/st-gorbatovski/sollama/src/sft/artifacts/tests/python_basics/rlhf/test-rlhf-LoRA_llama-7b-t_09-basics-mpnet_dot_prod_so_par-dot_prod-length_penalty-step_72-mean_reward_39.2-3_attempts-temp_0.7.csv --device cuda:2

# while ps -p 1836351 > /dev/null; do sleep 1; done; echo "Starting script at $(date)"; ./run.sh