import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--sample_number", type=int, default=0)
    parser.add_argument("--prompt", type=str, default='alpaca', help='wiz, alpaca')
    args = parser.parse_args()
    return args


def get_loss_part_text(tokenizer, messages, use_type, target_span, max_length, loss_list_):
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to('cpu')

    if use_type == "direct_answer_text":
        # direct_answer_text = "NPC: " + output_i
        text = messages[-1]["content"]
    
    elif use_type == "whole_text":
        # pass
        # whole_text = messages[:-2]["content"] + output_i
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            max_length=max_length,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    # input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
    start_index = text.rfind(target_span)
    text_temp = text[:start_index]
    token_id_temp = tokenizer.encode(text_temp)
    start_token = len(token_id_temp) 
    end_token_real = input_ids.shape[1]

    loss_list = loss_list_[start_token-1:end_token_real-1] 

    return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)


def main():

    args = parse_args()
    print(args)

    from transformers import AutoTokenizer, LlamaForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in range(len(pt_data)):
        # 这一层按道理得有一层变量
        tmp_mean_rate_list = []
        tmp_mean_list_1 = []
        tmp_mean_list_2 = []
        len_json = (len(json_data[i]['conversation'])-2)/2
        len_pt = len(pt_data[i])
        print(len_json, len_pt)
        print("===")

if __name__ == '__main__':
    main()