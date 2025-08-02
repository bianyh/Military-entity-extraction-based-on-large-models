import json
import random

# 读取 JSON 文件
with open('sample_entity.json', 'r') as f:
    sample_entity = json.load(f)

with open('sample_text.json', 'r') as f:
    sample_text = json.load(f)

# 获取所有 key
keys = list(sample_entity.keys())

# 设置随机种子（可选，用于保证结果可复现）
random.seed(42)

# 随机打乱 key 顺序
random.shuffle(keys)

# 划分比例（例如：验证集占 30%，训练集占 70%）
val_ratio = 0.3
split_idx = int(len(keys) * val_ratio)

val_keys = keys[:split_idx]
train_keys = keys[split_idx:]

# 创建训练集和验证集
train_entity = {k: sample_entity[k] for k in train_keys}
val_entity = {k: sample_entity[k] for k in val_keys}
train_text = {k: sample_text[k] for k in train_keys}
val_text = {k: sample_text[k] for k in val_keys}

# 保存到新的文件
with open('train_entity.json', 'w') as f:
    json.dump(train_entity, f, indent=2)

with open('val_entity.json', 'w') as f:
    json.dump(val_entity, f, indent=2)

with open('train_text.json', 'w') as f:
    json.dump(train_text, f, indent=2)

with open('val_text.json', 'w') as f:
    json.dump(val_text, f, indent=2)