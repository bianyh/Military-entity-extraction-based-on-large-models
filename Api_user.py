import os
import json
import torch
from tqdm import tqdm
import openai
from loguru import logger


class APIEntityExtractor:
    def __init__(self, api_key, model_name, api_base=None, default_headers=None, device="cuda"):
        """
        初始化实体提取器
        :param api_key: OpenAI API 密钥
        :param model_name: OpenAI 模型名称（例如 "gpt-3.5-turbo"）
        :param api_base: OpenAI API 的基础网址，默认为 None，如果不指定则使用 OpenAI 默认的公共 API
        :param default_headers: 默认的 HTTP 头信息
        :param device: 使用的设备，默认为 "cuda"，虽然这里不会用到 GPU，但保留参数以兼容旧代码
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.default_headers = default_headers if default_headers else {}
        self.device = device

        # 配置日志
        self.log_path = "./log"
        os.makedirs(self.log_path, exist_ok=True)
        logger.add(f"{self.log_path}/{os.path.basename(__file__)}.log", rotation="100 MB")

        # 设置 OpenAI API 密钥和基础网址
        openai.api_key = self.api_key
        if self.api_base:
            openai.base_url = self.api_base
        if self.default_headers:
            openai.default_headers = self.default_headers

    def inference_with_openai(self, text: str) -> list:
        """
        使用 OpenAI API 进行推理
        :param text: 输入文本
        :return: 提取的实体列表
        """
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in military information extraction."},
                    {"role": "user", "content": text}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response_text}\n")

            # ====== 尝试提取 JSON 部分 ======
            start = response_text.find("[")
            end = response_text.find("]") + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                entities = json.loads(json_str)
                return entities
            else:
                logger.warning("无法找到有效的 JSON 输出")
                return []

        except Exception as e:
            logger.warning(f"⚠️ 推理失败，返回空列表。错误信息：{e}")
            return []

    def get_raw_response_from_openai(self, text: str) -> str:
        """
        传入一段文本并返回 OpenAI API 的原始响应内容
        :param text: 输入文本
        :return: OpenAI API 的原始响应内容
        """
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response_text}\n")
            return response_text
        except Exception as e:
            logger.warning(f"⚠️ 获取原始响应失败。错误信息：{e}")
            return f"Error: Unable to get response from OpenAI API. Detailed error: {e}"

    def text_pre_judge(self, text: str, entity_name: str, entity_label: str) -> int:
        try:
            prompt = f"{text}\nPlease analyze whether this entity is a real and specific item that can appear in an image and be selected: {entity_name}\n If it can return 1, otherwise return 0. Some entities representing actions (such as cardiopulmonary resuscitation), or those representing a conceptual entity (such as a country or military unit), often cannot be reflected in the picture, so please boldly give 0. No explanation is needed."
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response_text}\n")
            return int(response_text)
        except Exception as e:
            logger.warning(f"⚠️ 获取原始响应失败。错误信息：{e}")
            return f"Error: Unable to get response from OpenAI API. Detailed error: {e}"

    def get_description_from_openai(self, text: str, entity_name: str) -> str:
        try:
            prompt = f"Please analyze what {entity_name} should look like from the following sentence to help identify it in the picture (from the surroundings, the actual shape, etc.)，And if you can analyze from the text that there may be similar entities in the diagram, you need to give some details to help accurately determine the entities:\n[{text}]\nNote that you need to combine this text to analyze the state of the {entity_name}！\nYou just need to give a short sentence describing what the {entity_name} might look like in the picture, without giving any explanation."
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response_text}\n")
            return response_text
        except Exception as e:
            logger.warning(f"⚠️ 获取原始响应失败。错误信息：{e}")
            return f"Error: Unable to get response from OpenAI API. Detailed error: {e}"

    def extract_entities_from_file(self, input_path: str, output_path: str):
        """
        从文件中提取实体并保存结果
        :param input_path: 输入文件路径，包含文本数据的 JSON 文件
        :param output_path: 输出文件路径，保存提取结果的 JSON 文件
        """
        if not os.path.exists(input_path):
            logger.error(f"❌ 找不到输入文件：{input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            texts = json.load(f)

        results = {}

        for idx, item in tqdm(texts.items(), desc="📍 正在抽取实体"):
            text = item["text"]
            try:
                logger.debug(f'处理文本{text}')
                prompt = f"""You are an expert in military information extraction. Your job is to extract ** Military-related named entities ** from military-related English text, and classify each entity into one of the following 6 types:
        ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]
        Below is a brief explanation of the 6 types:

        - Vehicle: A vehicle is a device or means of transportation designed for traveling on land. It can carry people or goods from one place to another, such as cars, buses, and bicycles.

        - Aircraft: An aircraft is a machine that can fly in the air. It generates thrust via engines and lift via wings to achieve flight, used for transporting passengers, cargo, or carrying out military missions.

        - Vessel: A vessel refers to a large - sized water - borne craft capable of navigating on water. It can be categorized into military vessels and civilian vessels, serving purposes such as maritime transportation, military operations, and scientific research.

        - Weapon: An offensive weapon is a device or instrument designed to attack targets and cause damage or destruction. Common examples include guns, missiles, and bombs, which play a crucial role in military operations.

        - Location: Location refers to the geographical position where something or somewhere is situated on the Earth's surface. It can be determined using geographical coordinates like latitude and longitude, and it's essential for describing the spatial distribution and connections of things.

        - Other: This category encompasses things or circumstances beyond the aforementioned concepts, exhibiting diversity and uncertainty, and requiring analysis based on specific contexts.

        Please follow these rules:
        1. Only extract entities that explicitly appear in the text and only classify them in the **6** types.
        2. Try to extract as little as possible that represents the entity, such as a codename. Only the most representative words can be extracted from the content that represents the same meaning in a sentence.
        3. You can only extract military-related entities with obvious codenames. Non-military entities, such as brushes, or military entities that do not explicitly indicate the code name, such as generic references such as "soldier","rifles",etc. do not need to be extracted.
        4. Use the exact JSON array format.
        5. Do not explain or describe the output. Just return pure JSON.
        6. If you are sure that this text does not have a military entity that needs to be extracted, please boldly return [].

        --- Examples ---

        Text: "The F-16 Fighting Falcon is an American fighter aircraft."
        Output:
        [
            {{"name":"F-16","label":"aircraft"}}
        ]

        Text: "The soldier is brushing his shoes with a brush"
        Output:
        []

        --- Now extract entities from the following text ---

        Text: "{text}"
        Output:
        """
                entities = self.inference_with_openai(prompt)
                results[idx] = entities
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"[Error] ID: {idx}, Error: {e}")
                results[idx] = []

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ 实体抽取完成，结果已保存至：{output_path}")

    def extract_entities_from_text(self, text: str) -> list:
        """
        直接传入文本，返回提取的实体 JSON 数据
        :param text: 输入文本
        :return: 提取的实体 JSON 数据
        """
        print(f"提取实体：{text}")
        prompt = f"""You are an expert in military information extraction. Your job is to extract ** Military-related named entities ** from military-related English text, and classify each entity into one of the following 6 types:
        ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]
        Below is a brief explanation of the 6 types:

        - Vehicle: A vehicle is a device or means of transportation designed for traveling on land. It can carry people or goods from one place to another, such as cars, buses, and bicycles.

        - Aircraft: An aircraft is a machine that can fly in the air. It generates thrust via engines and lift via wings to achieve flight, used for transporting passengers, cargo, or carrying out military missions.

        - Vessel: A vessel refers to a large - sized water - borne craft capable of navigating on water. It can be categorized into military vessels and civilian vessels, serving purposes such as maritime transportation, military operations, and scientific research.

        - Weapon: An offensive weapon is a device or instrument designed to attack targets and cause damage or destruction. Common examples include guns, missiles, and bombs, which play a crucial role in military operations.

        - Location: Location refers to the geographical position where something or somewhere is situated on the Earth's surface. It can be determined using geographical coordinates like latitude and longitude, and it's essential for describing the spatial distribution and connections of things.

        - Other: This category encompasses things or circumstances beyond the aforementioned concepts, exhibiting diversity and uncertainty, and requiring analysis based on specific contexts.

        Please follow these rules:
        1. Only extract entities that explicitly appear in the text and only classify them in the **6** types.
        2. Try to extract as little as possible that represents the entity, such as a codename. Only the most representative words can be extracted from the content that represents the same meaning in a sentence.
        3. You can only extract military-related entities with obvious codenames. Non-military entities, such as brushes, or military entities that do not explicitly indicate the code name, such as generic references such as "soldier","rifles",etc. do not need to be extracted.
        4. Use the exact JSON array format.
        5. Do not explain or describe the output. Just return pure JSON.
        6. If you are sure that this text does not have a military entity that needs to be extracted, please boldly return [].

        --- Examples ---

        Text: "The F-16 Fighting Falcon is an American fighter aircraft."
        Output:
        [
            {{"name":"F-16","label":"aircraft"}}
        ]

        Text: "The soldier is brushing his shoes with a brush"
        Output:
        []

        --- Now extract entities from the following text ---

        Text: "{text}"
        Output:
        """
        entities = self.inference_with_openai(prompt)
        return entities

    def reflect_text_entities(self, text, entities):
        logger.info(f"现在开始对文本{text}提取结果为{entities}进行反思!")
        prompt = f"""
        """


# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置 OpenAI API 密钥、模型名称和基础网址
    api_key = ""  # 替换为你的 OpenAI API 密钥
    model_name = "gpt-3.5-turbo"  # 或者使用 "gpt-4" 等其他模型
    api_base = "https://xh.v1api.cc/v1/"  # 替换为你的 OpenAI API 基础网址
    default_headers = {"x-foo": "true"}  # 替换为你的默认头信息

    # 创建实体提取器实例
    extractor = APIEntityExtractor(api_key, model_name, api_base, default_headers)

    # 输入输出文件路径
    input_path = "./数据集/val_text.json"
    output_path = "output/text_val_results/api_3.5.json"

    # 执行实体提取
    extractor.extract_entities_from_file(input_path, output_path)

    # 测试直接传入文本提取实体
    # sample_text = "Soldiers carry out combat shooting in the mountains using rifles during inspection on the plateau."
    # extracted_entities = extractor.extract_entities_from_text(sample_text)
    # print(f"Extracted entities: {extracted_entities}")

    # # 测试获取原始响应内容
    # raw_response = extractor.get_raw_response_from_openai('hello!')
    # print(f"Raw response from OpenAI API:\n{raw_response}")