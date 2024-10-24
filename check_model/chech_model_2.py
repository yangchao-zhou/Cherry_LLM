import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM
from matplotlib.colors import ListedColormap, BoundaryNorm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

all_total_elements = 0
all_num_equal = 0

# Initialize the models
model1_name = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_12b_mistral_v2_0925"
# model1_name = "/mnt/data/ran.xiao/cloud/prepare_for_online/llama3_as_en_12b_mistral_v2_1012"
# model2_name = "/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/models/llama2_2050_linky_ziya_nemo_12b_1021"
# model2_name = "/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/models/llama3_as_nemo_en_sft_1021"
model1 = AutoModelForCausalLM.from_pretrained(model1_name, torch_dtype=torch.bfloat16).to(device)
# model2 = AutoModelForCausalLM.from_pretrained(model2_name, torch_dtype=torch.bfloat16).to(device)


# Define weight difference function
def calculate_weight_diff(model1_weight, model2_weight):
    return torch.abs(model1_weight - model2_weight).mean().item()

# Define weight equality function (element-wise comparison)
def calculate_weight_equal(model1_weight, model2_weight):
    global all_total_elements  # 声明使用全局变量
    global all_num_equal  # 声明使用全局变量
    # 使用逐元素比较，返回相等元素的比例
    equal_elements = torch.eq(model1_weight, model2_weight)  # 返回逐元素相等的布尔张量
    num_equal = equal_elements.sum().item()  # 统计相等的元素数量
    total_elements = model1_weight.numel()  # 计算总的元素数量
    all_total_elements += total_elements
    all_num_equal += num_equal
    return num_equal / total_elements  # 返回相等元素占比


# Calculate layer differences
def calculate_layer_diffs(model1, model2):
    global all_total_elements  # 声明使用全局变量
    global all_num_equal  # 声明使用全局变量
    layer_diffs = []
    layer_equals = []

    for model1_layer, model2_layer in zip(model1.model.layers, model2.model.layers):
        layer_equal = {
            "self_attn_q_proj": calculate_weight_equal(
                model1_layer.self_attn.q_proj.weight, model2_layer.self_attn.q_proj.weight
            ),
            "self_attn_k_proj": calculate_weight_equal(
                model1_layer.self_attn.k_proj.weight, model2_layer.self_attn.k_proj.weight
            ),
            "self_attn_v_proj": calculate_weight_equal(
                model1_layer.self_attn.v_proj.weight, model2_layer.self_attn.v_proj.weight
            ),
            "self_attn_o_proj": calculate_weight_equal(
                model1_layer.self_attn.o_proj.weight, model2_layer.self_attn.o_proj.weight
            ),
            'input_layernorm': calculate_weight_equal(model1_layer.input_layernorm.weight, model2_layer.input_layernorm.weight),
            "mlp_down_proj": calculate_weight_equal(
                model1_layer.mlp.down_proj.weight, model2_layer.mlp.down_proj.weight
            ),
            "mlp_gate_proj": calculate_weight_equal(
                model1_layer.mlp.gate_proj.weight, model2_layer.mlp.gate_proj.weight
            ),
            "mlp_up_proj": calculate_weight_equal(
                model1_layer.mlp.up_proj.weight, model2_layer.mlp.up_proj.weight
            ),
            'post_attention_layernorm': calculate_weight_equal(model1_layer.post_attention_layernorm.weight, model2_layer.post_attention_layernorm.weight),
        }

        layer_diff = {
            "self_attn_q_proj": calculate_weight_diff(
                model1_layer.self_attn.q_proj.weight, model2_layer.self_attn.q_proj.weight
            ),
            "self_attn_k_proj": calculate_weight_diff(
                model1_layer.self_attn.k_proj.weight, model2_layer.self_attn.k_proj.weight
            ),
            "self_attn_v_proj": calculate_weight_diff(
                model1_layer.self_attn.v_proj.weight, model2_layer.self_attn.v_proj.weight
            ),
            "self_attn_o_proj": calculate_weight_diff(
                model1_layer.self_attn.o_proj.weight, model2_layer.self_attn.o_proj.weight
            ),
            'input_layernorm': calculate_weight_diff(model1_layer.input_layernorm.weight, model2_layer.input_layernorm.weight),
            "mlp_down_proj": calculate_weight_diff(
                model1_layer.mlp.down_proj.weight, model2_layer.mlp.down_proj.weight
            ),
            "mlp_gate_proj": calculate_weight_diff(
                model1_layer.mlp.gate_proj.weight, model2_layer.mlp.gate_proj.weight
            ),
            "mlp_up_proj": calculate_weight_diff(
                model1_layer.mlp.up_proj.weight, model2_layer.mlp.up_proj.weight
            ),
            'post_attention_layernorm': calculate_weight_diff(model1_layer.post_attention_layernorm.weight, model2_layer.post_attention_layernorm.weight),
        }

        layer_diffs.append(layer_diff)
        layer_equals.append(layer_equal)
    
    same_parameters_ratio = all_num_equal / all_total_elements  # 计算相等权重比例

    return layer_diffs, layer_equals, same_parameters_ratio


# Visualize the layer differences and save the plot
def visualize_layer_diffs(layer_diffs, model2_name, save_path="layer_diffs_plot.png"):
    num_layers = len(layer_diffs)
    num_components = len(layer_diffs[0])

    fig, axs = plt.subplots(1, num_components, figsize=(24, 8))
    fig.suptitle(f"{model1_name} <> {model2_name}", fontsize=16)

    # 定义颜色映射和边界，从黄色到蓝色
    # colors = ['#ffffe0', '#b2f5a9', '#87ceeb', '#1f77b4']  # 渐变颜色
    colors = ['#ffffe0', '#87ceeb', '#1f77b4']  # 渐变颜色
    # bounds = [0, 0.0001, 0.0003]  # 分成四档
    bounds = [0, 0.0001, 0.00015, 0.0002]  # 切得更细的五档

    # bounds = [0, 1e-5, 1e-4]  # 分成四档
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    for i, component in enumerate(layer_diffs[0].keys()):
        component_diffs = [[layer_diff[component]] for layer_diff in layer_diffs]
        
        # 绘制热图，使用自定义颜色映射
        sns.heatmap(component_diffs, annot=True, fmt=".8f", cmap=cmap, norm=norm, ax=axs[i], cbar_kws={"shrink": 0.8})
        axs[i].set_title(component)
        axs[i].set_xlabel("Layer")
        axs[i].set_ylabel("Difference")
        axs[i].set_xticks([])
        axs[i].set_yticks(range(num_layers))
        axs[i].set_yticklabels(range(num_layers))
        axs[i].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_layer_same_persent(layer_equals, model2_name, save_path="layer_equals_plot.png"):
    num_layers = len(layer_equals)
    num_components = len(layer_equals[0])

    fig, axs = plt.subplots(1, num_components, figsize=(24, 8))
    fig.suptitle(f"{model1_name} <> {model2_name}", fontsize=16)

    for i, component in enumerate(layer_equals[0].keys()):
        # 获取每一层对应组件的差异
        component_equals = [[layer_equal[component]*100] for layer_equal in layer_equals]  # 乘以100转换为百分比

        sns.heatmap(component_equals, annot=True, fmt=".2f", cmap="YlGnBu", ax=axs[i], 
                    cbar_kws={"shrink": 0.8}, yticklabels=range(num_layers))  # 确保y轴标签与层数对应
        axs[i].set_title(component)
        axs[i].set_xlabel("Layer")
        axs[i].set_ylabel("Equal present(%)") 
        axs[i].set_xticks([])

    # 将y轴标签反转，确保从顶部到底部对应层数
    axs[i].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_subfolder_paths(folder_path):
    # 使用 os.listdir() 获取文件夹中的所有条目，os.path.isdir() 判断是否为文件夹
    subfolders = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, entry))]
    return subfolders

def check_model():
    subfolder_paths = get_subfolder_paths("/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/models")
    for model2_name in subfolder_paths:
        model2 = AutoModelForCausalLM.from_pretrained(model2_name, torch_dtype=torch.bfloat16).to(device)
        # Calculate and save the differences
        layer_diffs, layer_equals, same_parameters_ratio = calculate_layer_diffs(model1, model2)
        print(f"参数完全相同的比例: {same_parameters_ratio}")

        visualize_layer_diffs(layer_diffs, model2_name, save_path="/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/pic-0925-all/"+ model1_name.split("/")[-1] + "对比" + model2_name.split("/")[-1]+ ".png")
        visualize_layer_same_persent(layer_equals, model2_name, save_path="/mnt/workspace/yangchao.zhou/opt/Cherry_LLM/check_model/pic-0925-all/equal/"+ model1_name.split("/")[-1] + "对比" + model2_name.split("/")[-1]+ ".png")
        print(f"done: {model2_name}")
    print("all done")

if __name__ == '__main__':
    check_model()