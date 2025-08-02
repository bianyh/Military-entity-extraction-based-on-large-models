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

    # 存储每个样本的详细信息
    detailed_results = {}

    for id in true_data:
        true_entities = true_data[id]
        pred_entities = pred_data.get(id, [])

        # 记录真实实体的匹配状态
        true_entities_matched = [False] * len(true_entities)

        # 记录预测实体是否正确
        pred_entities_correct = [False] * len(pred_entities)

        # 统计真实实体的总数
        total_true_entities += len(true_entities)
        total_pred_entities += len(pred_entities)

        # 存储当前样本的分析结果
        analysis = {
            'correct': [],
            'incorrect': [],
            'missed': []
        }

        # 检查每个预测实体是否匹配到真实实体
        for i, pred_entity in enumerate(pred_entities):
            pred_name = pred_entity['name']
            pred_label = pred_entity['label']
            pred_bnd = pred_entity.get('bnd', None)

            # 遍历真实实体，寻找匹配
            for j, true_entity in enumerate(true_entities):
                true_name = true_entity['name']
                true_label = true_entity['label']
                true_bndbox = true_entity.get('bnd', None)

                # 检查实体名称和类型是否匹配
                if pred_name == true_name and pred_label == true_label:
                    # 检查视觉区域是否匹配
                    if true_bndbox is None:
                        if pred_bnd is None:
                            # 实体不可定位，预测也为None，匹配成功
                            if not true_entities_matched[j]:
                                true_entities_matched[j] = True
                                pred_entities_correct[i] = True
                                true_positives += 1
                                analysis['correct'].append({
                                    'pred': pred_entity,
                                    'true': true_entity
                                })
                    else:
                        if pred_bnd is not None:
                            iou = compute_iou(list(pred_bnd.values()), list(true_bndbox.values()))
                            if iou > 0.5:
                                # 实体可定位，IoU>0.5，匹配成功
                                if not true_entities_matched[j]:
                                    true_entities_matched[j] = True
                                    pred_entities_correct[i] = True
                                    true_positives += 1
                                    analysis['correct'].append({
                                        'pred': pred_entity,
                                        'true': true_entity
                                    })

            # 如果预测实体不正确，记录为错误
            if not pred_entities_correct[i]:
                analysis['incorrect'].append(pred_entity)

        # 记录未召回的实体
        for j, matched in enumerate(true_entities_matched):
            if not matched:
                analysis['missed'].append(true_entities[j])

        detailed_results[id] = analysis
        # print(id)
        # print(analysis['incorrect'])
        # print(analysis['missed'])
        pass

    # 计算指标
    precision = true_positives / total_pred_entities if total_pred_entities > 0 else 0
    recall = true_positives / total_true_entities if total_true_entities > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, detailed_results

import os
print("保存路径:", os.getcwd())
# 文件路径
true_file_path = './数据集/val_entity.json'
pred_file_path = './output/results_reflectflow3.json'
text_path = './数据集/val_text.json'
with open(text_path, 'r', encoding='utf-8') as f:
    text_data = json.load(f)

# 评估结果
precision, recall, f1, detailed_results = evaluate_results(true_file_path, pred_file_path)

with open("./output/analysis_results.txt", "w", encoding="utf-8") as file:
    for id, analysis in detailed_results.items():
        if analysis['incorrect'] == [] and analysis['missed'] == []:
            continue
        file.write(f"\n{text_data[id]}\n")
        file.write(f"\nSample ID: {id}\n")
        file.write("\nCorrect Entities:\n")
        for item in analysis['correct']:
            file.write(f"  Predicted: {item['pred']}\n")
            file.write(f"  True: {item['true']}\n")

        file.write("\nIncorrect Entities (Predicted but not correct):\n")
        for entity in analysis['incorrect']:
            file.write(f"  {entity}\n")

        file.write("\nMissed Entities (True but not predicted):\n")
        for entity in analysis['missed']:
            file.write(f"  {entity}\n")

        file.write("\n" + "-" * 50 + "\n")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# workflow2:
# Precision: 0.8519
# Recall: 0.7667
# F1 Score: 0.8070

# workflow1
# Precision: 0.7821
# Recall: 0.6778
# F1 Score: 0.7262

# workflow3:
# Precision: 0.7957
# Recall: 0.8222
# F1 Score: 0.8087

# main_workflow
# Precision: 0.7158
# Recall: 0.7556
# F1 Score: 0.7351


# xiaorong1
# Precision: 0.7162
# Recall: 0.5889
# F1 Score: 0.6463
# 2
# Precision: 0.7660
# Recall: 0.8000
# F1 Score: 0.7826
# 3
# Precision: 0.7629
# Recall: 0.8222
# F1 Score: 0.7914
# 4
# Precision: 0.8395
# Recall: 0.7556
# F1 Score: 0.7953

