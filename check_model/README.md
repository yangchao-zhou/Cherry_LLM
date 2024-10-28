## 通过oss下载模型的方法

### 2050实验室线上模型的下载llama2_2050_linky_ziya_nemo_12b_1021
mkdir llama2_2050_linky_ziya_nemo_12b_1021 && cd llama2_2050_linky_ziya_nemo_12b_1021
ossutil64 cp -r --config ../ossconfig oss://sai-mnt/common/shengying.wei/2050/llama2_2050_linky_ziya_nemo_12b_1021/ .

### 林旗盛的模型下载llama3_as_en_12b_mistral_v2_0925
mkdir llama3_as_en_12b_mistral_v2_0925 && cd llama3_as_en_12b_mistral_v2_0925
ossutil64 cp -r --config ../ossconfig oss://sai-mnt/common/shengying.wei/ai_story/llama3_as_en_12b_mistral_v2_0925/ .

### 林旗盛的模型下载llama3_as_en_12b_mistral_v2_1012
mkdir llama3_as_en_12b_mistral_v2_1012 && cd llama3_as_en_12b_mistral_v2_1012
ossutil64 cp -r --config ../ossconfig oss://sai-mnt/common/shengying.wei/ai_story/llama3_as_en_12b_mistral_v2_1012/ .


### 张立昌的模型下载llama3_as_nemo_en2_sft_1021
mkdir llama3_as_nemo_en2_sft_1021 && cd llama3_as_nemo_en2_sft_1021
ossutil64 cp -r --config ../ossconfig oss://sai-mnt/common/shengying.wei/2050/llama3_as_nemo_en2_sft_1021/ .