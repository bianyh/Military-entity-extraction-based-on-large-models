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
api_base = APIconfig['api_base']  # 替换为你的 OpenAI API 基础网址
default_headers = {"x-foo": "true"}  # 替换为你的默认头信息
# 创建实体提取器实例
APIextractor = APIEntityExtractor(api_key, model_name, api_base, default_headers)

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
        logger.error(f"找不到输入文件：{input_path}")
        return

    logger.info("开始实体抽取流程")
    # ========== 开始流程处理 ==========
    results = {}
    for idx, item in tqdm(texts.items(), desc="正在抽取实体"):

        text = item["text"]
        try:
            logger.info(f"处理样本 {idx}: {text}")
            # ========== 首先提取文本实体 ==========
            try:
                entities = TextEntityExtractor.send_request(text)
                if entities == []:
                    results[idx] = []
                    logger.success(f"完成对{idx}样本的处理")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f'提取样本{idx}时候出现问题,转使用API提取:{e}')
                entities = APIextractor.extract_entities_from_text(text=text)
            logger.info(f"提取到的实体: {entities}")
            # ========== 剔除提取出来的相同名称和label实体，因为相同实体如果存在多次可以在后续图片中找到 ==========
            entities = remove_duplicate_entities(entities)
            if entities==[]:
                results[idx] = []
                logger.success(f"完成对{idx}样本的处理")

            # ========== 对图片实体进行框选（如果有图片） ==========
            image = str(idx) + '.jpg'
            if image in image_files:
                num = 0
                while num < len(entities):
                    entity = entities[num]
                    num += 1
                    if entity["label"] == "location":
                        logger.info("location will not appear in image!")
                        entity['bnd'] = None
                        if not results.__contains__(idx):
                            results[idx] = []
                        results[idx].append(entity)
                        continue

                    # ========== 使用API对所提取出的实体进行描述 ==========
                    description = APIextractor.get_description_from_openai(text=text, entity_name=entity["name"])
                    # ========== 组装prompt并使用视觉模型在图片中提取实体 ==========
                    #
                    #
                    # 先提取数量，再提取对应的图片实体内容。如何对文本实体的提取进行反思成为关键问题。
                    #
                    #
                    number_ref = 0
                    correct_ref = 0
                    reflect_text = ''
                    Number_of_errors = 0
                    prompt = ImageEntityExtractor.generate_prompt(text, entity["name"], entity["label"], description)
                    bnd = ImageEntityExtractor.extract_entities_from_image(image_input_path + image, prompt + reflect_text)
                    reflect_text = ''
                    index = 0
                    # ========== 对图片中提取出来的实体进行反思行为 ==========
                    yuan_entity = copy.deepcopy(entity)
                    for i in bnd:
                        entity = copy.deepcopy(yuan_entity)
                        index += 1
                        # 进行实体正确性的反思
                        entity['bnd'] = i['bnd']
                        correct_ref = ImageEntityExtractor.reflect_correct(text, image_path=image_input_path + image, entity=entity, description=description)
                        if correct_ref == 0:
                            reflect_text = f'{entity} is error!'
                            logger.error(reflect_text)
                            entity["bnd"] = "null"
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
                        logger.info(f"Extracted entities: {entities}")
                        if not results.__contains__(idx):
                            results[idx] = []
                        results[idx].append(entity)
                        logger.success(f"完成对{idx}样本的处理")
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Error] ID: {idx}, Error: {e}")
            results[idx] = []

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.success(f"实体抽取完成，结果已保存至：{output_path}")


if __name__ == "__main__":
    main()