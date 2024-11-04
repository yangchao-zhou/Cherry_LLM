import json
import os
from tqdm import tqdm

data_path = "../data/V15_2_struct.json"
save_path = data_path.replace(".json", "_process.jsonl")
f_s = open(save_path, 'w')

save_cnt = 0
save_datas = []
with open(data_path, 'r') as f:
    datas = json.load(f)
    for i, data in enumerate(tqdm(datas)):
        cur_id = "1031{:04d}".format(i)
        conversation = data["conversation"]
        assert conversation[0]["role_type"] == "system"
        assert conversation[1]["role_type"] == "for_refer_BOT"
        new_conversation = [conversation[0], conversation[1]]
        round_id = 1

        if "BOT" in conversation[1]["role_type"] and "BOT" in conversation[2]["role_type"]:
            start_idx = 3
        else:
            start_idx = 2
        if len(conversation[start_idx:]) % 2 != 0:
            continue

        for idx in range(start_idx, len(conversation), 2):
            assert conversation[idx]["role_type"] == "user"
            assert "BOT" in conversation[idx + 1]["role_type"]
            new_conversation.append(conversation[idx])
            new_conversation.append(conversation[idx + 1])
            cur_new_conversation = new_conversation
            save_data = {"id": cur_id,
                            "round_id": round_id,
                            "conversation": cur_new_conversation
                            }
            f_s.write(json.dumps(save_data, ensure_ascii=False) + "\n")
            f_s.flush()
            save_cnt += 1
            round_id += 1

print(save_cnt)





