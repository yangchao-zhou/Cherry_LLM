import json
import os

data_path = "../data/V15_2_struct_process_cherry.jsonl"

id_dict = {}
with open(data_path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        cur_id = data["id"]

        if cur_id not in id_dict:
            id_dict[cur_id] = []
        id_dict[cur_id].append(data)

print(len(id_dict))

id_dict = {k: v for k, v in sorted(id_dict.items(), key=lambda x: x[0])}
save_datas = []
round_err_cnt = 0
for cur_id, datas in id_dict.items():
    cur_datas = sorted(datas, key=lambda x: x["round_id"])
    round_list = []
    for idx, data in enumerate(cur_datas):
        round_id = data["round_id"]
        round_list.append(round_id)
        conversation = data["conversation"]
        ifd_score = data["ifd_score"]
        if idx == 0:
            assert len(conversation) % 2 == 0
            assert conversation[0]["role_type"] == "system"
            assert conversation[1]["role_type"] == "for_refer_BOT"
            save_conversation = [conversation[0], conversation[1]]
        else:
            assert conversation[-2]["role_type"] == "user"
            assert "BOT" in conversation[-1]["role_type"]
            save_conversation.append(conversation[-2])
            conversation[-1]["ifd_score"] = ifd_score
            save_conversation.append(conversation[-1])

    flag = False
    for i in range(1, len(round_list)):
        if round_list[i] != round_list[i-1] + 1:
            flag = True
            break
    if not flag and len(round_list) == max(round_list):
        save_datas.append({"conversation": save_conversation})
    else:
        round_err_cnt += 1
        # print("round err...")
print(round_err_cnt)



save_path = data_path.replace(".jsonl", "_restore.json")
with open(save_path, "w", encoding="utf-8") as fw:
    json.dump(save_datas, fw, indent=4, ensure_ascii=False)

