from EntityExtractor import EntityExtractor
from ImageEntityExtractor import ImageEntityExtractor
from Api_user import APIEntityExtractor
import os
import json
from tqdm import tqdm
from loguru import logger
import copy

# 配置 loguru 日志
logger.add("./log/app.log", format="{time} {level} {message}", level="INFO")

with open("./config.json", "r") as f:
    configs = json.load(f)


# 配置 loguru 日志
logger.add("./log/app.log", format="{time} {level} {message}", level="INFO")

# ========== TextEntityExtractor ==========
text_config = configs['TextEntityExtractor']
base_model_path = text_config['base_model_path']  # 基础模型路径
lora_adapter_path = text_config['lora_adapter_path']  # LoRA 适配器保存路
TextEntityExtractor = EntityExtractor(base_model_path, lora_adapter_path)

# ========== ImageEntityExtractor ==========
# 配置路径
MODEL_PATH = configs['ImageEntityExtractor']['MODEL_PATH']
# 创建图像实体提取器实例
ImageEntityExtractor = ImageEntityExtractor(MODEL_PATH)

# ========== APIEntityExtractor ==========


# 配置 OpenAI API 密钥、模型名称和基础网址
APIconfig = configs['APIEntityExtractor']
api_key = APIconfig['api_key']  # 替换为你的 OpenAI API 密钥
model_name = APIconfig['model_name']  # 或者使用 "gpt-4" 等其他模型
api_base = APIconfig['https://xh.v1api.cc/v1/']  # 替换为你的 OpenAI API 基础网址
default_headers = {"x-foo": "true"}  # 替换为你的默认头信息
# 创建实体提取器实例
APIextractor = APIEntityExtractor(api_key, model_name, api_base, default_headers)


# ========== 主程序 ==========
def main():
    DATA_CONFIG = configs["dataset"]
    # ========== 数据读入 ==========
    input_path = DATA_CONFIG["input_path"]
    image_input_path = DATA_CONFIG["image_input_path"]
    image_output_path = DATA_CONFIG["image_output_path"]

    with open(input_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    # 创建一个空列表来存储图片文件路径
    image_files = set()
    # 遍历文件夹中的所有文件
    for file in os.listdir(image_input_path):
        image_files.add(file)

    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件：{input_path}")
        return

    # ========== 开始流程处理 ==========
    results = {}
    for idx, item in tqdm(texts.items(), desc="正在抽取实体"):
        text = item["text"]
        try:
            print(text)
            # ========== 首先提取文本实体 ==========
            try:
                entities = TextEntityExtractor.send_request(text)
                if entities == []:
                    raise "提取出来个空的"
            except Exception as e:
                print(f'提取样本{idx}时候出现问题,转使用API提取。{e}')
                entities = APIextractor.extract_entities_from_text(text=text)
            print(entities)
            # ========== 对图片实体进行框选（如果有图片） ==========
            image = str(idx) + '.jpg'
            if image in image_files:
                for entity in entities:
                    # ========== 使用API对所提取出的实体进行描述 ==========
                    # description = APIextractor.get_description_from_openai(text=text, entity_name=entity["name"])
                    description = ''
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
                            cache = {}
                            cache["xmin"] = entity["bnd"][0]
                            cache["ymin"] = entity["bnd"][1]
                            cache["xmax"] = entity["bnd"][2]
                            cache["ymax"] = entity["bnd"][3]
                            entity["bnd"] = cache
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

    print(f"\n✅ 实体抽取完成，结果已保存至：{output_path}")


if __name__ == "__main__":
    main()