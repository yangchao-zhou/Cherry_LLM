import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index], add_special_tokens=False))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses


def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model.eval()

    if args.save_path[-3:] != '.pt':
        args.save_path += '.pt'
    # if os.path.exists(args.save_path):
    #     print('save_path exists!')
    #     raise Exception

    if args.data_path.endswith(".json"):
        with open(args.data_path, "r") as f:
            data = json.load(f)
    else:
        data = [json.loads(line.strip()) for line in open(args.data_path, "r")]

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    import time
    strat_time = time.time()
    new_data = []
    for i in tqdm(range(len(sampled_data))):
        data_i = sampled_data[i]
        conversation = data_i["conversation"]
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
            assert "BOT" in conversation[cur_idx+1]["role_type"]
            messages += format_dict["user_fmt"].format(conversation[cur_idx]["content"])
            messages += format_dict["ai_fmt"].format(conversation[cur_idx+1]["content"])
        messages += format_dict["user_fmt"].format(conversation[-2]["content"])
        instruct_i = messages + format_dict["ai_start_fmt"]
        messages += format_dict["ai_fmt"].format(conversation[-1]["content"])
        whole_text = messages
        direct_answer_text = format_dict["ai_fmt"].format(conversation[-1]["content"])
        output_i = conversation[-1]["content"]


        temp_data_i = {}
        if args.mod == 'pre':
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, instruct_i, args.max_length)
            temp_data_i['ppl'] = [ppl_ins_alone,0,0]
            temp_data_i['sent_emb'] = [emb_ins_alone,0,0]

        elif args.mod == 'cherry':
            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length, add_special_tokens=False).to(device)
            instruct_i_len = instruct_i_input_ids.shape[1] 
        
            ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, args.max_length-instruct_i_len+4)
            ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

            temp_data_i['ppl'] = [0,ppl_out_alone,ppl_out_condition]
            temp_data_i['token_loss'] = [[],loss_list_alone,loss_list_condition]

        new_data.append(temp_data_i)
        pass

    print('New data len:', len(new_data))
    save_path = args.save_path.replace(".pt", "_s{}_e{}.pt".format(start_idx, end_idx))
    torch.save(new_data, save_path)

    print('Time Used:', (time.time()-strat_time)/60, '(min)')

if __name__ == "__main__":
    main()