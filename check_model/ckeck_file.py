import hashlib

def calculate_md5(file_path):
    # 初始化 MD5 哈希对象
    md5_hash = hashlib.md5()
    
    # 以二进制方式打开文件，并逐块读取更新哈希
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    # 返回文件的 MD5 值
    return md5_hash.hexdigest()

# 示例：计算两个模型文件的 MD5
model1_path = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_12b_mistral_v2_1012/tokenizer_config.json"
model2_path = "/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/models/llama2_2050_rp_v2_character_f_1021/tokenizer_config.json"

md5_model1 = calculate_md5(model1_path)
md5_model2 = calculate_md5(model2_path)

print(f"MD5 of Model 1: {md5_model1}")
print(f"MD5 of Model 2: {md5_model2}")
