import json
import random as random_gen
from datasets import Dataset
import tqdm

def process_and_push(data_path, dataset_name, dataset_id):
    # 读取 JSON 文件

    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            data_list.append(json.loads(line))
    
    new_data = []
    for sample in tqdm.tqdm(data_list, desc=f"Processing {dataset_name}"):

        # 构建新样本，不包含 output 字段
        new_data.append({
            "messages": sample["messages"],
            "ground_truth": sample["ground_truth"],
            "dataset": "math",
        })
    
    print(f"📊 Processed {len(new_data)} samples")
    
    # 打乱顺序
    random_gen.shuffle(new_data)
    
    # 创建 Hugging Face Dataset
    dataset = Dataset.from_list(new_data)
    
    # 上传到 Hugging Face Hub
    dataset.push_to_hub(f"TTTXXX01/{dataset_name}")
    print(f"✅ Dataset {dataset_name} uploaded to Hugging Face Hub")

# 调用示例
process_and_push("/weka/oe-adapt-default/tengx/Data/filter_MathSub/MathSub_filtered.json", "MathSub_Filtered_LC", "math")
process_and_push("/weka/oe-adapt-default/tengx/Data/filter_MathSub/Omega_filtered.json", "Omega_Filtered_LC", "math")
process_and_push("/weka/oe-adapt-default/tengx/Data/filter_MathSub/Orz_filtered.json", "Orz_Filtered_LC", "math")