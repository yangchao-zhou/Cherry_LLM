1. 处理xiaoran训练格式的数据（standard_2_training.py后的数据），处理成单轮数据
运行process_data/process.py

2. 改cherry.sh的配置，注意配置start_idx和end_idx，多进程处理快一些

3. 运行运行process_data/merge_pt.py，合并pt文件

4. 运行IFD.sh

5. 运行process_data/restore_data.py，将单轮数据还原为原来多轮的数据格式

6. 运行sample_mistral.py或者sample_llama.py，进行比例采样（大于1的过滤，采样小于1的数据）