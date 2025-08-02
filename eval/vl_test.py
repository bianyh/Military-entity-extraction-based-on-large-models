# from EntityExtractor_request import EntityExtractor
from ImageEntityExtractor import ImageEntityExtractor
from Api_user import APIEntityExtractor
import os
import json
from tqdm import tqdm
import copy

# # ========== TextEntityExtractor ==========
# input_path = "./数据集/sample_text_filter.json"  # 输入数据集文件
# output_path = "./output/results.json"  # 输出结果文件
# api_url = "http://localhost:5000/extract_entities"  # Flask 后端接口地址
# TextEntityExtractor = EntityExtractor(input_path, output_path, api_url)

# ========== ImageEntityExtractor ==========
# 配置路径
MODEL_PATH = './models/qwen2.5-vl-7B'
# 创建图像实体提取器实例
ImageEntityExtractor = ImageEntityExtractor(MODEL_PATH)

# ========== APIEntityExtractor ==========
# 配置 OpenAI API 密钥、模型名称和基础网址
api_key = ""  # 替换为你的 OpenAI API 密钥
model_name = "gpt-3.5-turbo"  # 或者使用 "gpt-4" 等其他模型
api_base = "https://xh.v1api.cc/v1/"  # 替换为你的 OpenAI API 基础网址
default_headers = {"x-foo": "true"}  # 替换为你的默认头信息
# 创建实体提取器实例
APIextractor = APIEntityExtractor(api_key, model_name, api_base, default_headers)


# ========== 主程序 ==========
def main():
    # ========== 数据读入 ==========
    input_path = "./数据集/val_text.json"
    image_input_path = "./数据集/sample_image/"
    ans_path = "数据集/val_entity.json"
    image_output_path = './output/image/'
    output_path = './output/results.json'

    with open(input_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(ans_path, "r", encoding="utf-8") as f:
        ans = json.load(f)
    # 创建一个空列表来存储图片文件路径
    image_files = set()
    # 遍历文件夹中的所有文件
    for file in os.listdir(image_input_path):
        image_files.add(file)

    if not os.path.exists(input_path):
        print(f"找不到输入文件：{input_path}")
        return

    # ========== 开始流程处理 ==========
    results = {}
    for idx, item in tqdm(texts.items(), desc="正在抽取实体"):
        text = item["text"]
        try:
            print(text)
            entities = []
            for i in ans[idx]:
                entity = {}
                entity["name"] = i["name"]
                entity["label"] = i["label"]
                entities.append(entity)
            print(entities)
            if entities==[]:
                results[idx] = []
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            # ========== 对图片实体进行框选（如果有图片） ==========
            image = str(idx) + '.jpg'
            if image in image_files:
                for entity in entities:
                    # ========== 使用API对所提取出的实体进行描述 ==========
                    description = APIextractor.get_description_from_openai(text=text, entity_name=entity["name"])
                    # ========== 组装prompt并使用视觉模型在图片中提取实体 ==========
                    prompt = ImageEntityExtractor.generate_prompt(text, entity["name"], entity["label"], description)
                    bnd = ImageEntityExtractor.extract_entities_from_image(image_input_path + image, prompt)
                    index = 0
                    yuan_entity = copy.deepcopy(entity)
                    for i in bnd:
                        entity = copy.deepcopy(yuan_entity)
                        index += 1
                        entity['bnd'] = i['bnd']

                        if entity["bnd"] != "null":
                            ImageEntityExtractor.annotate_and_save_image(image_input_path + image, entity,
                                                                         image_output_path + str(idx) + '-' + str(
                                                                             index) + '-' + entity["name"] + '.jpg')
                        else:
                            entity['bnd'] = None
                        print(f"Extracted entities: {entities}")
                        if not results.__contains__(idx):
                            results[idx] = []
                        results[idx].append(entity)
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Error] ID: {idx}, Error: {e}")
            results[idx] = []

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n实体抽取完成，结果已保存至：{output_path}")


if __name__ == "__main__":
    main()

# Qwen2.5-VL-7B-Instruct
# Precision: 0.5147
# Recall: 0.7778
# F1 Score: 0.6195

# Qwen2.5-VL-3B-Instruct
# Precision: 0.5914
# Recall: 0.6111
# F1 Score: 0.6011