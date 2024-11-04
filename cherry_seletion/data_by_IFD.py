import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# mistral_12b, llama3
model_format_dict = {
    "mistral": {
        'system_fmt': "<s>[INST]{}[/INST]",
        'user_fmt': "[INST]{}[/INST]",
        'ai_fmt': "{}</s>",
        'ai_start_fmt': "",
    },
    "llama3": {
        'system_fmt': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
        'user_fmt': "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
        'ai_fmt': "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
        'ai_start_fmt': "<|start_header_id|>assistant<|end_header_id|>\n\n",
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--sample_number", type=int, default=0)
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    if args.json_data_path.endswith(".json"):
        with open(args.json_data_path, "r") as f:
            json_data = json.load(f)
    else:
        json_data = [json.loads(line.strip()) for line in open(args.json_data_path, "r")]

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        json_data_i = json_data[i]
        conversation = json_data_i["conversation"]
        assert len(conversation) % 2 == 0
        assert conversation[0]["role_type"] == "system"
        assert conversation[1]["role_type"] == "for_refer_BOT"
        messages = ""
        if "mistral" in args.model_name_or_path.lower():
            format_dict = model_format_dict["mistral"]
            system_prompt = conversation[0]["content"] + "\n\n" + conversation[1]["content"]
            messages += format_dict["system_fmt"].format(system_prompt)
        else:
            format_dict = model_format_dict["llama3"]
            messages += format_dict["system_fmt"].format(conversation[0]["content"])
            messages += format_dict["ai_fmt"].format(conversation[1]["content"])

        for cur_idx in range(2, len(conversation) - 2, 2):
            assert conversation[cur_idx]["role_type"] == "user"
            assert "BOT" in conversation[cur_idx + 1]["role_type"]
            messages += format_dict["user_fmt"].format(conversation[cur_idx]["content"])
            messages += format_dict["ai_fmt"].format(conversation[cur_idx + 1]["content"])
        messages += format_dict["user_fmt"].format(conversation[-2]["content"])
        instruct_i = messages + format_dict["ai_start_fmt"]
        messages += format_dict["ai_fmt"].format(conversation[-1]["content"])
        whole_text = messages
        direct_answer_text = format_dict["ai_fmt"].format(conversation[-1]["content"])
        output_i = conversation[-1]["content"]

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp, add_special_tokens=False)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]
            if start_token == 0:
                loss_list = loss_list_[start_token:end_token_real - 1]
            else:
                loss_list = loss_list_[start_token - 1:end_token_real - 1]

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if args.max_length-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, args.max_length-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, args.max_length, loss_2_list)

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

    # full_set = set(range(200))
    # missing_numbers = sorted(full_set - set(mean_rate_list_id_sample))
    # print(missing_numbers)


    new_data = [json_data[idx] for idx in mean_rate_list_id_sample]
    print('New data len \n',len(new_data))
    # with open(args.json_save_path, "w") as fw:
    #     json.dump(new_data, fw, indent=4)
    with open(args.json_save_path, "w") as fw:
        for nd in new_data:
            fw.write(json.dumps(nd, ensure_ascii=False) + '\n')
            fw.flush()



if __name__ == '__main__':
    main()