import json

file_path = "/weka/oe-adapt-default/tengx/main/open-instruct/scripts/data/MathSub-30K_0_3_sglang.jsonl"

count = 0
data = []

# 逐行读取 JSONL 文件
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        data.append(obj)
        count += 1

print(f"总共有 {count} 条数据\n")

# 打印前 5 条内容作为示例
for i, item in enumerate(data[:1], start=1):
    print(f"第 {i} 条:")
    print(len(item["output"][0]))
    print(len(item["output"][1]))
    print(len(item["output"][2]))
    print(len(item["output"][3]))
    print(len(item["output"][4]))
    print((item["output"][4]))
    print((item["output"][3]))
    print("-" * 40)
