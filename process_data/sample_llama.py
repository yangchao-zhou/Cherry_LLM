import json
import os
import random
import emoji


folder = "../data"
data_path = "V15_2_struct_process_cherry_restore.json"

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

rate = 0.7

save_folder = os.path.join(folder, "llama_rate_{}_v2".format(rate))

# data_path = "shujuzu_labeled_struct_process_cherry_restore.json"
# rate 0.4 : score : 0.6
# data_path = "mage_labeled_struct_process_cherry_restore.json"
# rate 0.8 : score : 0.48
# data_path = "V15_2_struct_process_cherry_restore.json"
# rate 0.8 : score : 0.48
# data_path = "train.mid.v3_notall_struct_process_cherry_restore.json"
# rate 0.05 : score : 0.62

with open(os.path.join(folder, data_path), 'r') as f:
    datas = json.load(f)
    print("session cnt:", len(datas))

random.seed(1234)
random.shuffle(datas)
train_data = datas[:int(len(datas)*0.98)]
ifd_score_list = []
for td in train_data:
    conversation = td["conversation"]
    for conver in conversation:
        role_type = conver["role_type"]
        if role_type == "BOT":
            if conver["ifd_score"] < 1:
                ifd_score_list.append(conver["ifd_score"])

print("ifd_score_list:", len(ifd_score_list))
ifd_score_list = sorted(ifd_score_list, reverse=True)
rate_score = ifd_score_list[int(len(ifd_score_list)*rate)]
print("rate_score: ", rate_score)

val_data = datas[int(len(datas)*0.98):]


def format_datas(datas, rate_score, is_train=True):
    round_cnt = 0
    struct_cnt = 0
    nostruct_cnt = 0
    origin_round_cnt = 0
    fm_datas = []
    for data in datas:
        conversation = data["conversation"]
        updated_conversation = []
        assert conversation[0]["role_type"] == "system"
        assert conversation[1]["role_type"] == "for_refer_BOT"
        system_prompt = conversation[0]["content"]
        if "npc_pic" in system_prompt:
            struct_cnt += 1
        else:
            nostruct_cnt += 1
        for conver in conversation[1:]:
            role_type = conver["role_type"]
            content = conver["content"]
            content = remove_emojis(content)
            if role_type == "user":
                value = content
                _from = "User"
                is_mask = True
            elif role_type == "BOT":
                value = content
                _from = "Assistant"
                is_mask = False
                origin_round_cnt += 1
                if is_train:
                    ifd_score = conver["ifd_score"]
                    if ifd_score < rate_score or ifd_score >= 1:
                        is_mask = True
            else:  # for_refer_BOT
                value = content
                _from = "Assistant"
                is_mask = True
            turn = {"value": value,
                    "from": _from,
                    "is_mask": is_mask,
                    "label": None}
            updated_conversation.append(turn)

        mask_list = [turn["is_mask"] for turn in updated_conversation]
        if mask_list.count(True) == len(updated_conversation):
            # print("not train data...")
            continue
        round_cnt += mask_list.count(False)
        conversation_obj = {
            "conversations": updated_conversation,
            "system": system_prompt,
            "mask": "User",
            "type": "VALUE_TO_TEXT",
        }
        fm_datas.append(conversation_obj)
    if is_train:
        print("train data ...")
    else:
        print("val data ...")
    print("struct_cnt:", struct_cnt)
    print("nostruct_cnt: ", nostruct_cnt)
    print("origin_round_cnt: ", origin_round_cnt)
    print("train_round_cnt: ", round_cnt)
    print("---"*30)
    return fm_datas


fm_train_data = format_datas(train_data, rate_score, is_train=True)
fm_val_data = format_datas(val_data, rate_score, is_train=False)
print("train_session_cnt: ", len(fm_train_data))
print("val_session_cnt: ", len(fm_val_data))


if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_path = os.path.join(save_folder, data_path.replace(".json", "_sft_train_llama.jsonl"))
f_s = open(save_path, 'w')
for data in fm_train_data:
    f_s.write(json.dumps(data, ensure_ascii=False) + "\n")
    f_s.flush()

save_path = os.path.join(save_folder, data_path.replace(".json", "_sft_val_llama.jsonl"))
f_s = open(save_path, 'w')
for data in fm_val_data:
    f_s.write(json.dumps(data, ensure_ascii=False) + "\n")
    f_s.flush()

