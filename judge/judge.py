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

def evaluate_results(true_file, pred_file):
    with open(true_file, 'r', encoding='utf-8') as f:
        true_data = json.load(f)
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    correct_samples = 0
    total_samples = len(true_data)

    for id in true_data:
        true_entities = true_data[id]
        try:
            pred_entities = pred_data[id]
        except:
            continue
        if true_entities == [] and pred_entities == []:
            correct_samples += 1
            continue
        pred_entities = pred_data.get(id, [])

        # 如果预测和真实实体数量不一致，直接跳过
        if len(pred_entities) != len(true_entities):
            continue

        sample_correct = True

        # 创建一个列表来跟踪真实实体是否被正确预测
        true_entities_matched = [False] * len(true_entities)

        for i, true_entity in enumerate(true_entities):
            true_name = true_entity['name']
            true_label = true_entity['label']
            true_bndbox = true_entity.get('bnd', None)  # 如果没有bndbox字段，默认为None

            for pred_entity in pred_entities:
                pred_name = pred_entity['name']
                pred_label = pred_entity['label']
                pred_bnd = pred_entity.get('bnd', None)  # 如果没有bnd字段，默认为None

                # 检查实体名称和类型是否匹配
                if pred_name == true_name and pred_label == true_label:
                    # 检查边界框是否匹配
                    if true_bndbox is None:
                        if pred_bnd is None:
                            true_entities_matched[i] = True
                    else:
                        if pred_bnd is not None:
                            iou = compute_iou(list(pred_bnd.values()), list(true_bndbox.values()))
                            if iou > 0.5:
                                true_entities_matched[i] = True

        # 检查所有真实实体是否都被正确匹配
        if all(true_entities_matched) and true_entities!=[]:
            # print(str(id) + ':')
            # print(true_entities)
            # print(pred_entities)
            # print(len(true_entities))
            correct_samples += 1
        else:
            print(str(id) + ':')
            print(true_entities)
            print(pred_entities)
            print(len(true_entities))
    print(correct_samples)
    print(total_samples)
    precision = correct_samples / total_samples if total_samples > 0 else 0
    recall = correct_samples / total_samples if total_samples > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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
# 99
# 120
# Precision: 0.8250


# workflow1
# 30
# 120
# Precision: 0.2500

# workflow3:
# 101
# 120
# Precision: 0.8417

# main_workflow
# 32
# 120
# Precision: 0.2667


# xiaorong1
# 86
# 120
# Precision: 0.7167

# 2
# 94
# 120
# Precision: 0.7833

# 3
# 92
# 120
# Precision: 0.7667

# 4
# 95
# 120
# Precision: 0.7917