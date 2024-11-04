import torch
import json
import os
import re
pt_folder = "../data/pt"
pt_start = "V15_2_struct_process_cherry"
pt_paths = [path for path in os.listdir(pt_folder) if path.startswith(pt_start) and "merge" not in path]
print(pt_paths)
pt_paths = sorted(pt_paths, key=lambda x: int(re.findall(r'\d+', x)[-1]))

merge_pt_datas = []
for path in pt_paths:
    print(path)
    pt_data = torch.load(os.path.join(pt_folder, path), map_location=torch.device('cpu'))
    merge_pt_datas.extend(pt_data)
    print(len(pt_data))

print(len(merge_pt_datas))
save_path = os.path.join(pt_folder, "{}_merge.pt".format(pt_start))
torch.save(merge_pt_datas, save_path)
