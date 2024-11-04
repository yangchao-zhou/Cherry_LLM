CUDA_VISIBLE_DEVICES=0 python cherry_seletion/data_analysis.py \
    --data_path data/V15_2_struct_process.jsonl \
    --save_path data/pt/V15_2_struct_process_cherry.pt \
    --model_name_or_path /mnt/data/nlp_models/mistralai/Mistral-Nemo-Instruct-2407-ModifiedChatTemplate \
    --max_length 8192 \
    --start_idx 0 \
    --end_idx 1500 \
    --mod cherry
