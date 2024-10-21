from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

base_model_name = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_22b_mistral_v1_0925"
chat_model_name = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_22b_mistral_v1_0925"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype=torch.bfloat16)

def calculate_weight_diff(base_weight, chat_weight):
    return torch.abs(base_weight - chat_weight).mean().item()

def calculate_layer_diffs(base_model, chat_model):
    layer_diffs = []
    for base_layer, chat_layer in zip(base_model.model.layers, chat_model.model.layers):
        layer_diff = {
            'input_layernorm': calculate_weight_diff(base_layer.input_layernorm.weight, chat_layer.input_layernorm.weight),
            # 'mlp_down_proj': calculate_weight_diff(base_layer.mlp.down_proj.weight, chat_layer.mlp.down_proj.weight),
            # 'mlp_gate_proj': calculate_weight_diff(base_layer.mlp.gate_proj.weight, chat_layer.mlp.gate_proj.weight),
            # 'mlp_up_proj': calculate_weight_diff(base_layer.mlp.up_proj.weight, chat_layer.mlp.up_proj.weight),
            'post_attention_layernorm': calculate_weight_diff(base_layer.post_attention_layernorm.weight, chat_layer.post_attention_layernorm.weight),
            'self_attn_q_proj': calculate_weight_diff(base_layer.self_attn.q_proj.weight, chat_layer.self_attn.q_proj.weight),
            'self_attn_k_proj': calculate_weight_diff(base_layer.self_attn.k_proj.weight, chat_layer.self_attn.k_proj.weight),
            'self_attn_v_proj': calculate_weight_diff(base_layer.self_attn.v_proj.weight, chat_layer.self_attn.v_proj.weight),
            'self_attn_o_proj': calculate_weight_diff(base_layer.self_attn.o_proj.weight, chat_layer.self_attn.o_proj.weight)
        }
        layer_diffs.append(layer_diff)
    return layer_diffs

def visualize_layer_diffs(layer_diffs):
    num_layers = len(layer_diffs)
    num_components = len(layer_diffs[0])

    fig, axs = plt.subplots(1, num_components, figsize=(24, 8))
    fig.suptitle(f"{base_model_name} <> {chat_model_name}", fontsize=16)

    for i, component in enumerate(layer_diffs[0].keys()):
        component_diffs = [[layer_diff[component]] for layer_diff in layer_diffs]
        sns.heatmap(component_diffs, annot=True, fmt=".6f", cmap="YlGnBu", ax=axs[i], cbar_kws={"shrink": 0.8})
        axs[i].set_title(component)
        axs[i].set_xlabel("Layer")
        axs[i].set_ylabel("Difference")
        axs[i].set_xticks([])
        axs[i].set_yticks(range(num_layers))
        axs[i].set_yticklabels(range(num_layers))
        axs[i].invert_yaxis()

    plt.tight_layout()
    plt.show()

layer_diffs = calculate_layer_diffs(base_model, chat_model)

visualize_layer_diffs(layer_diffs)