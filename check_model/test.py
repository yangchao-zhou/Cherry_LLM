from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 指定模型文件的路径
model_dir = "/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/llama2_2050_rp_v2_minor_protect_1021"

# 加载模型和分词器，使用 .bin 文件
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 测试生成文本
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
