# import json
#
# def evaluate_entity_extraction(predicted_file, gold_file):
#     # 加载预测结果和正确答案
#     with open(predicted_file, 'r', encoding='utf-8') as f:
#         predicted_data = json.load(f)
#     with open(gold_file, 'r', encoding='utf-8') as f:
#         gold_data = json.load(f)
#
#     cuowu = {}
#
#     correct = 0  # 正确预测的实体数目
#     predicted_total = 0  # 预测实体的总数目
#     gold_total = 0  # 真实存在的实体数目
#
#     # 遍历每个数据ID
#     for data_id in predicted_data:
#         # 获取预测的实体列表和真实的实体列表
#         predicted_entities = predicted_data.get(data_id, [])
#         gold_entities = gold_data.get(data_id, [])
#
#         # 统计预测和真实的实体数目
#         predicted_total += len(predicted_entities)
#         gold_total += len(gold_entities)
#
#         # 如果预测实体和真实实体的数量不一致，跳过该数据
#         if len(predicted_entities) != len(gold_entities):
#             continue
#
#         # 检查所有预测实体是否都与真实实体匹配
#         match = True
#         for pred_entity in predicted_entities:
#             matched = False
#             for gold_entity in gold_entities:
#                 if pred_entity['name'] == gold_entity['name'] and pred_entity['label'] == gold_entity['label']:
#                     matched = True
#                     break
#             if not matched:
#                 match = False
#                 break
#
#         # 如果所有预测实体都匹配，则认为预测正确
#         if match:
#             correct += len(predicted_entities)
#         else:
#             cuowu[data_id] = gold_entities
#             cuowu[data_id+'err'] = predicted_entities
#
#     # 计算精确率、召回率和F1值
#     precision = correct / predicted_total if predicted_total > 0 else 0
#     recall = correct / gold_total if gold_total > 0 else 0
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#
#     # 将字典写入JSON文件
#     with open("./output/error_entities.json", 'w', encoding='utf-8') as f:
#         json.dump(cuowu, f, ensure_ascii=False, indent=4)
#
#     return precision, recall, f1_score
#
# # 示例用法
# predicted_file = './output/api_4o.json'  # 替换为你的预测结果文件路径
# gold_file = './数据集/val_entity.json'  # 替换为正确的答案文件路径
#
# precision, recall, f1 = evaluate_entity_extraction(predicted_file, gold_file)
#
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1-Score: {f1:.4f}')









import json


def remove_duplicate_entities(entities):
    seen = set()
    unique_entities = []
    for entity in entities:
        # 以name和label组合成元组作为唯一标识
        identifier = (entity['name'], entity['label'])
        if identifier not in seen:
            seen.add(identifier)
            unique_entities.append(entity)
    return unique_entities

def evaluate_entity_extraction(predicted_file, gold_file):
    # 加载预测结果和正确答案
    with open(predicted_file, 'r', encoding='utf-8') as f:
        predicted_data = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    cuowu = {}
    label_stats = {}  # 新增：按label统计的字典

    correct = 0  # 正确预测的实体数目
    predicted_total = 0  # 预测实体的总数目
    gold_total = 0  # 真实存在的实体数目

    # 遍历每个数据ID
    for data_id in predicted_data:
        # 获取预测的实体列表和真实的实体列表
        predicted_entities = predicted_data.get(data_id, [])
        gold_entities = gold_data.get(data_id, [])

        predicted_entities = remove_duplicate_entities(predicted_entities)
        gold_entities = remove_duplicate_entities(gold_entities)

        # 统计预测和真实的实体数目
        predicted_total += len(predicted_entities)
        gold_total += len(gold_entities)

        # 创建一个标记列表，用于标记哪些真实实体已经被匹配
        matched_gold = [False] * len(gold_entities)

        # 遍历预测实体，检查是否与真实实体匹配
        for i, pred_entity in enumerate(predicted_entities):
            matched = False
            for j, gold_entity in enumerate(gold_entities):
                # 如果实体名称和标签都匹配，并且该真实实体尚未被匹配
                if (pred_entity['name'] == gold_entity['name'] and
                    pred_entity['label'] == gold_entity['label'] and
                    not matched_gold[j]):
                    matched = True
                    matched_gold[j] = True
                    correct += 1  # 增加正确实体计数

                    # 更新按label统计的正确数
                    label = pred_entity['label']
                    if label not in label_stats:
                        label_stats[label] = {'correct': 0, 'predicted': 0, 'gold': 0}
                    label_stats[label]['correct'] += 1
                    break

            if not matched:
                # 如果预测实体未匹配，记录错误信息
                if data_id not in cuowu:
                    cuowu[data_id] = {
                        'gold_entities': gold_entities,
                        'predicted_entities': []
                    }
                cuowu[data_id]['predicted_entities'].append(pred_entity)

                # 更新按label统计的预测数（即使未匹配也计数）
                label = pred_entity['label']
                if label not in label_stats:
                    label_stats[label] = {'correct': 0, 'predicted': 0, 'gold': 0}
                label_stats[label]['predicted'] += 1

        # 记录未被匹配的真实实体
        for j, gold_entity in enumerate(gold_entities):
            if not matched_gold[j]:
                if data_id not in cuowu:
                    cuowu[data_id] = {
                        'gold_entities': gold_entities,
                        'predicted_entities': predicted_entities
                    }

                # 更新按label统计的真实数
                label = gold_entity['label']
                if label not in label_stats:
                    label_stats[label] = {'correct': 0, 'predicted': 0, 'gold': 0}
                label_stats[label]['gold'] += 1

    print(f"correct entities: {correct}")
    print(f"Predicted total entities: {predicted_total}")
    print(f"Gold total entities: {gold_total}")

    # 计算总体的精确率、召回率和F1值
    precision = correct / predicted_total if predicted_total > 0 else 0
    recall = correct / gold_total if gold_total > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算每个label的精确率、召回率和F1值
    label_metrics = {}
    for label in label_stats:
        correct = label_stats[label]['correct']
        predicted = label_stats[label]['predicted']
        gold = label_stats[label]['gold']

        label_precision = correct / predicted if predicted > 0 else 0
        label_recall = correct / gold if gold > 0 else 0
        label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall) if (label_precision + label_recall) > 0 else 0

        label_metrics[label] = {
            'precision': label_precision,
            'recall': label_recall,
            'f1': label_f1,
            'correct': correct,
            'predicted': predicted,
            'gold': gold
        }

    # 将错误信息写入JSON文件
    with open("./output/error_entities.json", 'w', encoding='utf-8') as f:
        json.dump(cuowu, f, ensure_ascii=False, indent=4)

    # 将按label统计的结果写入JSON文件
    with open("./output/label_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(label_metrics, f, ensure_ascii=False, indent=4)

    # # 打印每个label的指标
    # print("\nLabel-wise Metrics:")
    # for label in label_metrics:
    #     metrics = label_metrics[label]
    #     print(f"\nLabel: {label}")
    #     print(f"  Precision: {metrics['precision']:.4f}")
    #     print(f"  Recall: {metrics['recall']:.4f}")
    #     print(f"  F1-Score: {metrics['f1']:.4f}")
    #     print(f"  Correct: {metrics['correct']}")
    #     print(f"  Predicted: {metrics['predicted']}")
    #     print(f"  Gold: {metrics['gold']}")

    return precision, recall, f1_score, label_metrics

# 示例用法
predicted_file = "./output/text_val_results/api_3.5.json"  # 替换为你的预测结果文件路径
# predicted_file = "./output/results_reflectflow3.json"
gold_file = './数据集/val_entity.json'  # 替换为正确的答案文件路径

precision, recall, f1, label_metrics = evaluate_entity_extraction(predicted_file, gold_file)

print(f'Overall Metrics:')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')




# 4o
# correct entities: 55
# Predicted total entities: 64
# Gold total entities: 90
# Overall Metrics:
# Precision: 0.8594
# Recall: 0.6111
# F1-Score: 0.7143

# 4o_mini
# correct entities: 51
# Predicted total entities: 57
# Gold total entities: 90
# Overall Metrics:
# Precision: 0.8947
# Recall: 0.5667
# F1-Score: 0.6939

# 3.5
# correct entities: 47
# Predicted total entities: 55
# Gold total entities: 90
# Overall Metrics:
# Precision: 0.8545
# Recall: 0.5222
# F1-Score: 0.6483

# correct entities: 71
# Predicted total entities: 74
# Gold total entities: 90
# Overall Metrics:
# Precision: 0.9595
# Recall: 0.7889
# F1-Score: 0.8659

# correct entities: 65
# Predicted total entities: 68
# Gold total entities: 90
# Overall Metrics:
# Precision: 0.9559
# Recall: 0.7222
# F1-Score: 0.8228