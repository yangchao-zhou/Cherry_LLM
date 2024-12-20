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
    for i in tqdm(range(len(pt_data))):
        # 这一层按道理得有一层变量
        tmp_mean_rate_list = []
        tmp_mean_list_1 = []
        tmp_mean_list_2 = []
        for j in range(len(pt_data[i])):
            pt_data_i_j = pt_data[i][j]
            loss_1_list = pt_data_i_j['token_loss'][1]
            loss_2_list = pt_data_i_j['token_loss'][2]

            info = json_data[i]['conversation'][j]

            if info["role_type"] == "system":
                system_content = info["content"]
            elif info["role_type"] == "for_refer_BOT":
                messages = [
                    {"role":"system",
                    "content": system_content + '\ncharacter greeting:\n' + info["content"]
                    }
                ]
            elif info["role_type"] == "user":
                messages.append(
                    {"role":"user",
                    "content": info["content"]
                    }
                )
            elif info["role_type"] == "BOT":
                messages.append(
                    {"role":"assistant",
                    "content": info["content"]
                    }
                )
                
                # Tokenize the input text
                instruct_i_input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

                instruct_i_len = instruct_i_input_ids.shape[1]
                
                output_i = messages[-1]["content"]

                if args.max_length-instruct_i_len > 0:

                    len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, messages, "direct_answer_text", output_i, args.max_length-instruct_i_len+3, loss_1_list)
                    len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, messages, "whole_text", output_i, args.max_length, loss_2_list)

                    if len_1 <= 0 or len_2 <= 0:
                        continue

                    if instruct_i_len + len_1 > args.max_length:
                        continue

                    mean_1 = loss_list_1.mean()
                    mean_2 = loss_list_2.mean()
                    mean_rate = mean_2/mean_1
                    if mean_rate > 1: 
                        continue

                    mean_rate_list.append((mean_rate,i))
                    mean_list_1.append((mean_1,i))
                    mean_list_2.append((mean_2,i))

                else:
                    continue

    print('Do Rate')
    mean_rate_list = sorted(mean_rate_list)
    if args.sample_number == 0:
        args.sample_number = int(len(mean_rate_list)*args.sample_rate)
    mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
    mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]
    mean_rate_list_id_sample = sorted(mean_rate_list_id_sample)

    new_data = [json_data[idx] for idx in mean_rate_list_id_sample]
    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)


if __name__ == '__main__':
    main()