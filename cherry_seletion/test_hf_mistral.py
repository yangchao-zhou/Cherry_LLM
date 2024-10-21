from transformers import AutoTokenizer, AutoConfig
model_path = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_22b_mistral_v1_0925"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
chat = [
{"role": "system", "content": "you are a assistant"},
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

input_ids = tokenizer.apply_chat_template(chat, tokenize=False)
print(input_ids)