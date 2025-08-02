import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger


class EntityExtractor:
    def __init__(self, base_model_path, lora_adapter_path, device="cuda"):
        """
        初始化实体提取器
        :param base_model_path: 基础模型路径
        :param lora_adapter_path: LoRA 适配器保存路径
        :param device: 使用的设备，默认为 "cuda"，如果不可用则自动切换到 "cpu"
        """
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device if torch.cuda.is_available() else "cpu"

        # 配置日志
        self.log_path = "./log"
        os.makedirs(self.log_path, exist_ok=True)
        logger.add(f"{self.log_path}/{os.path.basename(__file__)}.log", rotation="100 MB")

        # 加载基础模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )

        # 加载 LoRA 适配器到基础模型上
        self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
        self.model.eval()  # 设置为评估模式

    def send_request(self, text: str, max_new_tokens: int = 64) -> list:
        """
        使用微调后的 LoRA 模型进行推理
        :param text: 输入文本
        :param max_new_tokens: 最大生成新 token 数量，默认为 256
        :return: 提取的实体列表
        """
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
        3. You can only extract entities of military significance with obvious codenames. Non-military entities, such as brushes, or military entities that do not explicitly indicate the code name, such as generic references such as "soldier","rifles",etc. do not need to be extracted.
        4. Use the exact JSON array format.
        5. Do not explain or describe the output. Just return pure JSON.
        6. If you are sure that this text does not have a military entity that needs to be extracted, please return [].

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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model.base_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self.model.base_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_start_str = "Output:\n"
        output_start_index = response.rfind(output_start_str)
        if output_start_index != -1:
            response = response[output_start_index + len(output_start_str):].strip()
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response.strip()}\n")

            # ====== 尝试提取 JSON 部分 ======
            try:
                # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
                start = response.find("[")
                end = response.find("]") + 1
                json_str = response[start:end]

                # Step 2: 正常解析 JSON
                entities = json.loads(json_str)
                ans = []
                for entity in entities:
                    if entity["name"] not in text:
                        continue
                    else:
                        ans.append(entity)
                return ans

            except Exception as e:
                logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
                return []
        else:
            logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response.strip()}\n")

            # ====== 尝试提取 JSON 部分 ======
            try:
                # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
                start = response.find("[")
                end = response.find("]") + 1
                json_str = response[start:end]

                # Step 2: 正常解析 JSON
                entities = json.loads(json_str)
                ans = []
                for entity in entities:
                    if entity["name"] not in text:
                        continue
                    else:
                        ans.append(entity)
                return ans

            except Exception as e:
                logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
                return []

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
                entities = self.inference_with_lora(text)
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
        entities = self.inference_with_lora(text)
        return entities


# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置路径
    base_model_path = "./models/chatglm3-6b"  # 基础模型路径
    lora_adapter_path = "./output/sft_lora_chatglm3/lora_adapters"  # LoRA 适配器保存路径

    # 创建实体提取器实例
    extractor = EntityExtractor(base_model_path, lora_adapter_path)

    # 输入输出文件路径
    input_path = "./数据集/sample_text_filter.json"
    output_path = "./output/entity_text_only_finetuned.json"

    # 执行实体提取
    extractor.extract_entities_from_file(input_path, output_path)

    # 测试直接传入文本提取实体
    sample_text = "The Ki-30, also known as the Type 97 Attack Bomber, along with the Type 98 Attack Bomber, was a main dive bomber of the Imperial Japanese Army."
    extracted_entities = extractor.extract_entities_from_text(sample_text)
    print(f"Extracted entities: {extracted_entities}")

# from flask import Flask, request, jsonify
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from loguru import logger

# # 配置日志
# log_path = "./log"
# os.makedirs(log_path, exist_ok=True)
# logger.add(f"{log_path}/app.log", rotation="100 MB")  # 自动分割日志文件

# # 配置路径
# base_model_path = "./models/glm3-6b"  # 基础模型路径
# lora_adapter_path = "./output/sft_lora_glm3/checkpoint-2240"  # LoRA 适配器保存路径
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 加载基础模型和分词器
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     trust_remote_code=True,
#     torch_dtype=torch.float16
# ).to(device)

# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_path,
#     trust_remote_code=True
# )

# # 加载 LoRA 适配器到基础模型上
# model = PeftModel.from_pretrained(model, lora_adapter_path)
# model.eval()  # 设置为评估模式

# app = Flask(__name__)


# # 推理示例函数
# def inference_with_lora(text: str, max_new_tokens: int = 256) -> str:
#     """
#     使用微调后的 LoRA 模型进行推理
#     """
#     prompt = f"""You are an expert in military information extraction. Your job is to extract ** Military-related named entities ** from military-related English text, and classify each entity into one of the following 6 types:
#         ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]
#         Below is a brief explanation of the 6 types:

#         - Vehicle: A vehicle is a device or means of transportation designed for traveling on land. It can carry people or goods from one place to another, such as cars, buses, and bicycles.

#         - Aircraft: An aircraft is a machine that can fly in the air. It generates thrust via engines and lift via wings to achieve flight, used for transporting passengers, cargo, or carrying out military missions.

#         - Vessel: A vessel refers to a large - sized water - borne craft capable of navigating on water. It can be categorized into military vessels and civilian vessels, serving purposes such as maritime transportation, military operations, and scientific research.

#         - Weapon: An offensive weapon is a device or instrument designed to attack targets and cause damage or destruction. Common examples include guns, missiles, and bombs, which play a crucial role in military operations.

#         - Location: Location refers to the geographical position where something or somewhere is situated on the Earth's surface. It can be determined using geographical coordinates like latitude and longitude, and it's essential for describing the spatial distribution and connections of things.

#         - Other: This category encompasses things or circumstances beyond the aforementioned concepts, exhibiting diversity and uncertainty, and requiring analysis based on specific contexts.

#         Please follow these rules:
#         1. Only extract entities that explicitly appear in the text and only classify them in the **6** types.
#         2. Try to extract as little as possible that represents the entity, such as a codename. Only the most representative words can be extracted from the content that represents the same meaning in a sentence.
#         3. You can only extract entities of military significance with obvious codenames. Non-military entities, such as brushes, or military entities that do not explicitly indicate the code name, such as generic references such as "soldier","rifles",etc. do not need to be extracted.
#         4. Use the exact JSON array format.
#         5. Do not explain or describe the output. Just return pure JSON.
#         6. If you are sure that this text does not have a military entity that needs to be extracted, please return [].

#         --- Examples ---

#         Text: "The F-16 Fighting Falcon is an American fighter aircraft."
#         Output:
#         [
#             {{"name":"F-16","label":"aircraft"}}
#         ]

#         Text: "The soldier is brushing his shoes with a brush"
#         Output:
#         []

#         --- Now extract entities from the following text ---

#         Text: "{text}"
#         Output:
#     """
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     try:
#         with torch.no_grad():
#             outputs = model.base_model.generate(
#                 inputs.input_ids,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False
#             )
#     except torch.cuda.OutOfMemoryError:
#         torch.cuda.empty_cache()
#         with torch.no_grad():
#             outputs = model.base_model.generate(
#                 inputs.input_ids,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False
#             )

#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     output_start_str = "Output:\n"
#     output_start_index = response.rfind(output_start_str)
#     if output_start_index != -1:
#         response = response[output_start_index + len(output_start_str):].strip()
#         logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response.strip()}\n")

#         # ====== 尝试提取 JSON 部分 ======
#         try:
#             # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
#             start = response.find("[")
#             end = response.rfind("]") + 1
#             json_str = response[start:end]

#             # Step 2: 正常解析 JSON
#             entities = json.loads(json_str)
#             ans = []
#             for entity in entities:
#                 if entity["name"] not in text:
#                     continue
#                 else:
#                     ans.append(entity)
#             return ans

#         except Exception as e:
#             logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
#             return []
#     else:
#         logger.debug(f"\n🔍 [DEBUG] 模型原始输出:\n{response.strip()}\n")

#         # ====== 尝试提取 JSON 部分 ======
#         try:
#             # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
#             start = response.find("[")
#             end = response.rfind("]") + 1
#             json_str = response[start:end]

#             # Step 2: 正常解析 JSON
#             entities = json.loads(json_str)
#             ans = []
#             for entity in entities:
#                 if entity["name"] not in text:
#                     continue
#                 else:
#                     ans.append(entity)
#             return ans

#         except Exception as e:
#             logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
#             return []


# @app.route('/extract_entities', methods=['POST'])
# def extract_entities():
#     try:
#         data = request.get_json()
#         text = data.get('text')
#         if not text:
#             return jsonify({"error": "No text provided"}), 400

#         entities = inference_with_lora(text)
#         return jsonify(entities), 200

#     except Exception as e:
#         logger.error(f"Error processing request: {e}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# import json
# import requests
# import os
#
# # 配置文件路径
# input_path = "./数据集/sample_text_filter.json"  # 输入数据集文件
# output_path = "./output/results.json"  # 输出结果文件
# api_url = "http://localhost:5000/extract_entities"  # Flask 后端接口地址
#
# # 确保输出目录存在
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
# # 读取数据集
# if not os.path.exists(input_path):
#     print(f"找不到输入文件：{input_path}")
#     exit(1)
#
# with open(input_path, "r", encoding="utf-8") as f:
#     texts = json.load(f)
#
# # 初始化结果字典
# results = {}
#
# # 遍历数据集并发送请求
# for idx, item in texts.items():
#     text = item["text"]
#     print(f"处理 ID: {idx}, 文本: {text}")
#
#     # 发送 POST 请求到 Flask 后端
#     response = requests.post(api_url, json={"text": text})
#
#     # 检查响应状态
#     if response.status_code == 200:
#         entities = response.json()
#         results[idx] = entities
#         print(f"成功获取结果：{entities}")
#     else:
#         print(f"请求失败，状态码：{response.status_code}")
#         results[idx] = []
#
# # 保存结果到文件
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)
#
# print(f"所有数据处理完成，结果已保存到：{output_path}")