import json

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

from collections import defaultdict

def evaluate_results(true_file, pred_file):
    with open(true_file, 'r', encoding='utf-8') as f:
        true_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    correct_samples = 0
    total_samples = len(true_data)

    for id in true_data:
        true_entities = true_data[id]
        pred_entities = pred_data.get(id, [])

        if not true_entities and not pred_entities:
            correct_samples += 1
            continue

        # 分组真实实体：(name, label) -> list[entity]
        grouped_true = defaultdict(list)
        for ent in true_entities:
            grouped_true[(ent['name'], ent['label'])].append(ent)

        # 对每组找到最大面积的实体
        max_area_true_entities = {}
        for key, ents in grouped_true.items():
            max_ent = None
            max_area = -1
            for ent in ents:
                if ent.get('bnd') is None:
                    area = 0
                else:
                    b = ent['bnd']
                    area = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
                if area > max_area:
                    max_area = area
                    max_ent = ent
            max_area_true_entities[key] = max_ent

        # 对每一组，查看预测结果是否匹配到了面积最大的实体
        matched_keys = set()

        for pred_ent in pred_entities:
            pred_name = pred_ent['name']
            pred_label = pred_ent['label']
            pred_bnd = pred_ent.get('bnd', None)
            key = (pred_name, pred_label)

            if key not in max_area_true_entities:
                continue

            true_ent = max_area_true_entities[key]
            true_bnd = true_ent.get('bnd', None)

            if true_bnd is None and pred_bnd is None:
                matched_keys.add(key)
            elif true_bnd and pred_bnd:
                iou = compute_iou(list(pred_bnd.values()), list(true_bnd.values()))
                if iou > 0.5:
                    matched_keys.add(key)

        # 判断该样本是否所有实体组都成功匹配
        if set(max_area_true_entities.keys()) == matched_keys:
            correct_samples += 1
        else:
            print(f"{id}:")
            print("True Entities:", true_entities)
            print("Predicted Entities:", pred_entities)
            print("Matched Keys:", matched_keys)

    precision = correct_samples / total_samples if total_samples > 0 else 0
    recall = precision
    f1 = precision

    print(correct_samples)
    print(total_samples)
    return precision, recall, f1


# 文件路径
true_file_path = './数据集/val_entity.json'
pred_file_path = './output/main_workflow.json'

# 评估结果
precision, recall, f1 = evaluate_results(true_file_path, pred_file_path)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# workflow2:
# 107
# 120
# Precision: 0.8917


# workflow1
# 101
# 120
# Precision: 0.8417

# workflow3:
# 111
# 120
# Precision: 0.9250

# main_workflow
# 105
# 120
# Precision: 0.8750

# 1
# 94
# 120
# Precision: 0.7833
# 2
# 102
# 120
# Precision: 0.8500
# 3
# 105
# 120
# Precision: 0.8750
# 4
# 100
# 120
# Precision: 0.8333