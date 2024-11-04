python cherry_seletion/data_by_IFD_vic.py \
    --model_name_or_path /mnt/data/nlp_models/mistralai/Mistral-Nemo-Instruct-2407-ModifiedChatTemplate \
    --pt_data_path data/pt/V15_2_struct_process_cherry_merge.pt \
    --json_data_path data/V15_2_struct_process.jsonl \
    --json_save_path data/V15_2_struct_process_cherry.jsonl \
    --max_length 8192 \
    --sample_rate 1