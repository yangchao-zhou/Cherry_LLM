import os
import json
import torch
import argparse
import jinja2
from tqdm import tqdm
import torch.nn as nn

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer

'''
萧然的model /mnt/workspace/yangchao.zhou/opt/models/Mistral-Nemo-Instruct-2407
/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_22b_mistral_v1_0925

原始的model /mnt/data/models/Mistral-Nemo-Instruct-2407
'''
tmpl_path = "tmpl/nsfw_npc_sys_prompt_1010_struct.tmpl"
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_prompt_template(template_file):
    with open(template_file, 'r') as f:
        template = f.read()
    template = jinja2.Template(template)
    return template
system_prompt_template_struct = load_prompt_template(tmpl_path)

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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='wiz', help='wiz, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    print(args)
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    # decoded_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

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
    """获取文本的ppl 和 loss

    Args:
        tokenizer (_type_): _description_
        model (_type_): _description_
        text (_type_): _description_
        target_span (_type_): _description_
        max_length (_type_): _description_

    Returns:
        _type_: _description_
    """    

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
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


def get_perplexity_and_embedding_messages(tokenizer, model, messages, use_type, target_span, max_length):
    """获取文本的ppl 和 loss

    Args:
        tokenizer (_type_): _description_
        model (_type_): _description_
        text (_type_): _description_
        target_span (_type_): _description_
        max_length (_type_): _description_

    Returns:
        _type_: _description_
    """

    # 使用 tokenizer.apply_chat_template 编码 messages
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

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

    # 找到 target_span 在原始文本中的起始位置
    start_index = text.rfind(target_span)

    # 计算 start_token 位置，基于编码后的 input_ids
    start_token = len(tokenizer.encode(text[:start_index]))

    # 获取 input_ids 的总长度
    end_token = input_ids.shape[1]

    # 复制 input_ids 作为标签
    labels = input_ids.clone()

    # 将 start_token 之前的 token 标记为 -100
    labels[0, :start_token] = -100

    # 禁用梯度计算
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # 计算损失和困惑度
    loss = outputs.loss
    perplexity = torch.exp(loss)

    # 初始化损失列表
    losses = []
    logits = outputs.logits

    # 逐个 token 计算负对数似然损失
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i - 1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    # 返回困惑度、0 和各个 token 的损失
    return perplexity.to("cpu"), 0, losses

def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')
    model.eval()

    if os.path.exists(args.save_path):
        print('save_path exists!')
        # raise Exception

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    new_data = []
    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        conversation_temp_data_i = []
        
        intro = data_i["intro"]
        greeting = data_i['greeting']
        npc_name = data_i['npc_name']
        history = data_i['history']
        npc_profile = data_i['npc_profile']

        systemp_prompt = system_prompt_template_struct.render({"npc_name": npc_name, "intro": intro, "npc_profile": npc_profile})
  
        messages = [
            {"role": "system",
            "content": systemp_prompt + '\ncharacter greeting:\n' + greeting}
        ]
        
        for j, info in enumerate(history):
            chat_temp_data_i = {}
            
            if info["role"] == "user":
                messages.append(
                    {"role": "user",
                    "content": info["content"]}
                )
            elif info["role"] == "ai":
                messages.append(
                    {"role": "assistant",
                    "content": info["content"]}
                )

                instruct_i_input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                instruct_i_len = instruct_i_input_ids.shape[1]
                
                output_i = messages[-1]["content"]

                ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_messages(tokenizer, model, messages, "direct_answer_text", output_i, args.max_length-instruct_i_len+3)
                ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_messages(tokenizer, model, messages, "whole_text", output_i, args.max_length)

                # 添加 ppl 和 token_loss 到 history 中
                history[j]['ppl'] = [0, ppl_out_alone, ppl_out_condition]
                history[j]['token_loss'] = [[], loss_list_alone, loss_list_condition]

                chat_temp_data_i['ppl'] = [0, ppl_out_alone, ppl_out_condition]
                chat_temp_data_i['token_loss'] = [[], loss_list_alone, loss_list_condition]

                conversation_temp_data_i.append(chat_temp_data_i)
            else:
                raise Exception

        new_data.append(data_i)

    # 保存更新后的数据为新的 pt 文件
    torch.save(new_data, args.save_path)
    print('Data saved to PT:', args.save_path)


if __name__ == "__main__":
    main()