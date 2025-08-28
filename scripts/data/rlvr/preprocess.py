from datasets import Dataset, load_dataset
import tqdm
from datasets import load_dataset, Dataset
import random
import re


random_gen = random.Random(42)

def format_logic_puzzle(puzzle_text):

    formatted_text = puzzle_text.strip()
    

    formatted_text = re.sub(r'\?\?+', '?', formatted_text)
    
    def format_list_items(match):
        items = match.group(1)
        # 分割并添加反引号
        formatted_items = []
        for item in items.split(','):
            item = item.strip()
            if item:
                formatted_items.append(f'`{item}`')
        return ', '.join(formatted_items)
    
    patterns = [
        (r': ([a-z, ]+)(?=\n|$)', format_list_items),  # 匹配特征列表
    ]
    
    for pattern, replacement in patterns:
        formatted_text = re.sub(pattern, lambda m: ': ' + format_list_items(m), formatted_text)
    
    lines = formatted_text.split('\n')
    formatted_lines = []
    clues_added = False
    
    for line in lines:
        if re.match(r'^\d+\.', line.strip()) and not clues_added:
            formatted_lines.append('')  # 空行
            formatted_lines.append('## Clues:')
            clues_added = True
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def convert_question_to_messages(question_text):

    formatted_puzzle = format_logic_puzzle(question_text)

    return [
        {
            "role": "user",
            "content": formatted_puzzle
        }
    ]


def process_and_push(data_path, dataset_name,id):
    train_data = load_dataset(data_path, split="train", streaming=True)
        
    new_data = []
    for sample in tqdm.tqdm(train_data, desc=f"Processing {dataset_name}"):
        messages = convert_question_to_messages(sample["question"])
        
        # 确保messages中的content有真正的换行符
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].replace('\\n', '\n')
        
        new_data.append({
            "messages": messages,
            "ground_truth": sample["answer"],
            "metadata": sample["metadata"],
            "dataset": id,
        })
    
    print(f"📊 Processed {len(new_data)} samples")
    
    random_gen.shuffle(new_data)
    dataset = Dataset.from_list(new_data)
    

    dataset.push_to_hub(f"TTTXXX01/{dataset_name}", split="train")

process_and_push("TTTXXX01/Puzzle_Zebra_All", "Puzzle_Zebra_20K", "Zebra")











# dataset_configs = [
#     ("online_eval/logic__zebra_puzzle_dataset_200.parquet", "logic__zebra_puzzle_dataset_200","zebra"),
# ]

# # dataset_configs = [
# #     ("train/logic__barc_1.6k.parquet", "rlvr_logic__barc_1.6k"),
# #     ("train/logic__arcagi1_111.parquet", "rlvr_logic__arcagi1_111"),
# #     ("train/logic__arcagi2_190.parquet", "rlvr_logic__arcagi2_190"),
# #     ("train/logic__graph_logical_1.2k.parquet", "rlvr_logic__graph_logical_1.2k"),
# #     ("train/logic__ordering_puzzle_1.9k.parquet", "rlvr_logic__ordering_puzzle_1.9k"),
# #     ("train/logic__zebra_puzzle_1.3k.parquet", "rlvr_logic__zebra_puzzle_1.3k"),
# # ]


# for parquet_path, dataset_name, id in dataset_configs:
#     process_and_push(parquet_path, dataset_name, id)


# # from datasets import load_dataset, concatenate_datasets

# # def normalize_ground_truth(example):
# #     gt = example["ground_truth"]
# #     if isinstance(gt, dict):
# #         import json
# #         return {"ground_truth": json.dumps(gt)}
# #     else:
# #         return {"ground_truth": str(gt)}

# # subset_names = [
# #     "rlvr_logic__barc_1.6k",
# #     "rlvr_logic__zebra_puzzle_1.3k",
# #     "rlvr_logic__ordering_puzzle_1.9k",
# #     "rlvr_logic__graph_logical_1.2k",
# #     "rlvr_logic__arcagi2_190",
# #     "rlvr_logic__arcagi1_111",
# # ]

# # all_datasets = []
# # for name in subset_names:
# #     ds = load_dataset(f"TTTXXX01/{name}", split="train")
# #     ds = ds.map(normalize_ground_truth)
# #     all_datasets.append(ds)

# # merged_dataset = concatenate_datasets(all_datasets)
# # merged_dataset.push_to_hub("TTTXXX01/rlvr_logic", split="train")

from datasets import load_dataset, DatasetDict
import datasets
import os
# from datasets import Dataset, load_dataset
# import tqdm
# import random


# random_gen = random.Random(42)


# def process_and_push(parquet_path, dataset_name,id):
#     train_data = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)

#     new_data = []
#     for sample in tqdm.tqdm(train_data, desc=dataset_name):
#         new_data.append({
#             "messages": sample["prompt"],
#             "ground_truth": sample["reward_model"]["ground_truth"],
#             "dataset": id,
#         })

#     random_gen.shuffle(new_data)
#     dataset = Dataset.from_list(new_data)



#     dataset.push_to_hub(f"TTTXXX01/{dataset_name}", split="test")



# dataset_configs = [
#     ("online_eval/logic__zebra_puzzle_dataset_200.parquet", "logic__zebra_puzzle_dataset_200","zebra"),
# ]

# # dataset_configs = [
# #     ("train/logic__barc_1.6k.parquet", "rlvr_logic__barc_1.6k"),
# #     ("train/logic__arcagi1_111.parquet", "rlvr_logic__arcagi1_111"),
# #     ("train/logic__arcagi2_190.parquet", "rlvr_logic__arcagi2_190"),
# #     ("train/logic__graph_logical_1.2k.parquet", "rlvr_logic__graph_logical_1.2k"),
# #     ("train/logic__ordering_puzzle_1.9k.parquet", "rlvr_logic__ordering_puzzle_1.9k"),
# #     ("train/logic__zebra_puzzle_1.3k.parquet", "rlvr_logic__zebra_puzzle_1.3k"),
# # ]


# for parquet_path, dataset_name, id in dataset_configs:
#     process_and_push(parquet_path, dataset_name, id)


# # from datasets import load_dataset, concatenate_datasets

# # def normalize_ground_truth(example):
# #     gt = example["ground_truth"]
# #     if isinstance(gt, dict):
# #         import json
# #         return {"ground_truth": json.dumps(gt)}
# #     else:
# #         return {"ground_truth": str(gt)}

# # subset_names = [
# #     "rlvr_logic__barc_1.6k",
# #     "rlvr_logic__zebra_puzzle_1.3k",
# #     "rlvr_logic__ordering_puzzle_1.9k",
# #     "rlvr_logic__graph_logical_1.2k",
# #     "rlvr_logic__arcagi2_190",
# #     "rlvr_logic__arcagi1_111",
# # ]

# # all_datasets = []
# # for name in subset_names:
# #     ds = load_dataset(f"TTTXXX01/{name}", split="train")
# #     ds = ds.map(normalize_ground_truth)
# #     all_datasets.append(ds)

# # merged_dataset = concatenate_datasets(all_datasets)
# # merged_dataset.push_to_hub("TTTXXX01/rlvr_logic", split="train")

# from datasets import load_dataset, DatasetDict
# import datasets
# import os

# # 你要修改的这些数据集的名称（已从截图中提取）
# dataset_names = [
#     "TTTXXX01/rlvr_logic",
# ]



# for dataset_name in dataset_names:
#     print(f"Processing: {dataset_name}")
    
#     try:
#         # 加载 train split
#         dataset = load_dataset(dataset_name, split="test")
        
#         # 显式转换为字符串类型
#         dataset = dataset.map(lambda x: {"ground_truth": str(x["ground_truth"])})
        
#         # 上传到新的 repo（可以自定义新名字或原地覆盖）
#         dataset.push_to_hub(dataset_name)
#         print(f"✅ Uploaded: {dataset_name}")
    
#     except Exception as e:
#         print(f"❌ Failed to process {dataset_name}: {e}")


# from datasets import load_dataset, DatasetDict, Dataset
# from huggingface_hub import login
# import os

# dataset = load_dataset("TTTXXX01/rlvr_logic")

# def repeat_dataset(ds, repeat=5):
#     return Dataset.from_dict(ds.to_dict()).select([i % len(ds) for i in range(len(ds) * repeat)])

# repeated_dataset = DatasetDict()
# for split in dataset:
#     repeated_dataset[split] = repeat_dataset(dataset[split], repeat=5)

# repeated_dataset.push_to_hub("TTTXXX01/rlvr_logic_repeat")


# from datasets import load_dataset
# from huggingface_hub import login

# # --------- 配置区域 ---------
# # 原数据集名称
# source_dataset = "hamishivi/rlvr_orz_math_57k_collected"
# # 上传到你账号下的新数据集名称（确保前缀是你的用户名）
# target_dataset = "TTTXXX01/rlvr_orz_math_57k_copy"  # ⚠️ 改成你的用户名
# # 是否设置为私有数据集
# private = False
# # --------------------------------

# # 如果你没运行过 huggingface-cli login，可以在此处使用 token 登录：
# # login(token="hf_xxx")

# print(f"📥 Loading source dataset: {source_dataset}")
# dataset = load_dataset(source_dataset, split="train")

# print(f"📤 Uploading to: {target_dataset}")
# dataset.push_to_hub(
#     target_dataset,
#     private=private,
# )

# print("✅ 上传成功！可以在浏览器打开：")
# print(f"https://huggingface.co/datasets/{target_dataset}")

# from datasets import load_dataset

# # 设置目标路径
# save_path = "/weka/oe-adapt-default/tengx/huggingface/rlvr_orz_math_57k_collected"

# # 下载数据集（train 分片）
# dataset = load_dataset("hamishivi/rlvr_orz_math_57k_collected", split="train")

# # 保存到指定路径
# dataset.save_to_disk(save_path)

# print(f"✅ 数据集已保存到本地: {save_path}")

# from datasets import load_dataset
# dataset = load_dataset("hamishivi/rlvr_orz_math_57k_collected", split="train")
# dataset.push_to_hub("TTTXXX01/rlvr_orz_math_57k_copy")

# from huggingface_hub import HfApi, create_repo
# from datasets import Dataset
# import os

# # 初始化API
# api = HfApi()

# # 创建数据集仓库
# repo_id = "TTTXXX01/rlvr_orz_math_57k_copy"
# create_repo(repo_id, repo_type="dataset", private=False)  # 设置private=True如果需要私有

# # 上传整个文件夹
# api.upload_folder(
#     folder_path="/weka/oe-adapt-default/tengx/huggingface/rlvr_orz_math_57k_collected/",
#     repo_id=repo_id,
#     repo_type="dataset"
# )

# from datasets import load_from_disk
# from huggingface_hub import HfApi, create_repo
# # dataset = load_from_disk("/weka/oe-adapt-default/tengx/huggingface/rlvr_orz_math_57k_collected")
# # create_repo("TTTXXX01/rlvr_orz_math_57k_copy", repo_type="dataset", private=False)  # 设置private=True如果需要私有
# # dataset.push_to_hub("TTTXXX01/rlvr_orz_math_57k_copy")



# from huggingface_hub import hf_hub_download
# import shutil

# # 下载到缓存路径
# # cached_path = hf_hub_download(
# #     repo_id="hamishivi/rlvr_orz_math_57k_collected",
# #     repo_type="dataset",
# #     filename="data/train-00000-of-00001.parquet"
# # )

# # # 拷贝到你想要的地址
# # target_path = "./rain-00000-of-00001.parquet"
# # shutil.copy(cached_path, target_path)

# # print(f"Saved to: {target_path}")

# from datasets import Dataset, load_from_disk

# dataset = load_from_disk("./your_data")