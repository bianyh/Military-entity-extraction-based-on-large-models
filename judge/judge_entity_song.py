import json
from collections import defaultdict


def compute_iou(box1, box2):
    """
    计算两个边界框的交并比（IoU）。
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: IoU值
    """
    # 计算交集区域
    box1 = list(map(int, box1))
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def evaluate_results(true_file, pred_file):
    with open(true_file, 'r', encoding='utf-8') as f:
        true_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    true_positives = 0
    total_true_entities = 0
    total_pred_entities = 0

    detailed_results = {}

    for id in true_data:
        true_entities = true_data[id]
        pred_entities = pred_data.get(id, [])

        total_pred_entities += len(pred_entities)

        # Group real entities by (name, label)
        grouped_true = defaultdict(list)
        for ent in true_entities:
            key = (ent['name'], ent['label'])
            grouped_true[key].append(ent)

        total_true_entities += len(true_entities)

        analysis = {
            'correct': [],
            'incorrect': [],
            'missed': []
        }

        pred_used = [False] * len(pred_entities)
        matched_keys = set()  # 记录哪些(name, label)组已经被匹配

        for i, pred_ent in enumerate(pred_entities):
            pred_name = pred_ent['name']
            pred_label = pred_ent['label']
            pred_bnd = pred_ent.get('bnd', None)

            matched = False

            key = (pred_name, pred_label)
            if key in grouped_true and key not in matched_keys:
                # 获取真实组中面积最大的实体
                candidates = grouped_true[key]
                max_area = -1
                max_ent = None
                for ent in candidates:
                    if ent.get('bnd') is not None:
                        b = ent['bnd']
                        area = (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
                    else:
                        area = 0
                    if area > max_area:
                        max_area = area
                        max_ent = ent

                true_bnd = max_ent.get('bnd', None)

                if true_bnd is None and pred_bnd is None:
                    matched = True
                elif true_bnd is not None and pred_bnd is not None:
                    iou = compute_iou(list(pred_bnd.values()), list(true_bnd.values()))
                    if iou > 0.5:
                        matched = True

                if matched:
                    # 匹配成功，只记一次
                    matched_keys.add(key)
                    true_positives += len(candidates)  # 这里将该组中所有实体视为召回成功
                    pred_used[i] = True
                    analysis['correct'].append({
                        'pred': pred_ent,
                        'true_group': candidates
                    })
                    continue

            # 若未匹配
            analysis['incorrect'].append(pred_ent)

        # 没被预测覆盖的真实实体
        for key, ents in grouped_true.items():
            if key not in matched_keys:
                analysis['missed'].extend(ents)

        detailed_results[id] = analysis

    precision = true_positives / total_pred_entities if total_pred_entities > 0 else 0
    recall = true_positives / total_true_entities if total_true_entities > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, detailed_results



import os
print("保存路径:", os.getcwd())
# 文件路径
true_file_path = './数据集/val_entity.json'
pred_file_path = './output/main_workflow.json'
text_path = './数据集/val_text.json'
with open(text_path, 'r', encoding='utf-8') as f:
    text_data = json.load(f)

# 评估结果
precision, recall, f1, detailed_results = evaluate_results(true_file_path, pred_file_path)

# with open("./output/analysis_results.txt", "w", encoding="utf-8") as file:
#     for id, analysis in detailed_results.items():
#         file.write(f"\n{text_data[id]}\n")
#         file.write(f"\nSample ID: {id}\n")
#         file.write("\nCorrect Entities:\n")
#         for item in analysis['correct']:
#             file.write(f"  Predicted: {item['pred']}\n")
#             file.write(f"  True: {item['true']}\n")
#
#         file.write("\nIncorrect Entities (Predicted but not correct):\n")
#         for entity in analysis['incorrect']:
#             file.write(f"  {entity}\n")
#
#         file.write("\nMissed Entities (True but not predicted):\n")
#         for entity in analysis['missed']:
#             file.write(f"  {entity}\n")
#
#         file.write("\n" + "-" * 50 + "\n")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# workflow2:
# Precision: 0.9383
# Recall: 0.8444
# F1 Score: 0.8889

# workflow1
# Precision: 0.8718
# Recall: 0.7556
# F1 Score: 0.8095

# workflow3:
# Precision: 0.8710
# Recall: 0.9000
# F1 Score: 0.8852

# main_workflow
# Precision: 0.7895
# Recall: 0.8333
# F1 Score: 0.8108


# xiaorong1
# Precision: 0.6667
# Recall: 0.5333
# F1 Score: 0.5926
# 2
# Precision: 0.7160
# Recall: 0.6444
# F1 Score: 0.6784
# 3
# Precision: 0.7126
# Recall: 0.6889
# F1 Score: 0.7006
# 4
# Precision: 0.7179
# Recall: 0.6222
# F1 Score: 0.6667
