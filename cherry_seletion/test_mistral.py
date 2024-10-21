from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

mistral_models_path = "/mnt/data/models/Mistral-Nemo-Instruct-2407"
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path)

system = "You are a assistant named chaoyang"
prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."
target = "May vary depending on specific logistical and operational challenges."
query = "who are you "
# completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
completion_request = ChatCompletionRequest(messages=[SystemMessage(content=system), UserMessage(content=prompt), AssistantMessage(content=target),UserMessage(content=query)])
tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, logprobs = generate([tokens], model, max_tokens=64, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

print(result)

from transformers import Transformer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(mistral_models_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(mistral_models_path, cache_dir='../cache')
tokenizer.encode()