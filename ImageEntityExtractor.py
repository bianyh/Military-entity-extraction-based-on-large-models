import json
import re
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from loguru import logger

class ImageEntityExtractor:
    def __init__(self, model_path):
        """
        初始化图像实体提取器
        :param model_path: 模型路径
        """
        self.model_path = model_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def extract_entities_from_image(self, image_path: str, description: str) -> list:
        """
        从图片和描述文本中提取实体
        :param image_path: 图片路径
        :param description: 图片描述文本
        :return: 提取的实体 JSON 数据
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": description,
                    },
                ],
            }
        ]

        # 准备推理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0]

        try:
            # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
            start = response.find("[")
            end = response.rfind("]") + 1
            json_str = response[start:end]

            # Step 2: 正常解析 JSON
            entities = json.loads(json_str)
        except Exception as e:
            logger.warning(f"⚠json.loads 失败，返回空列表。错误信息：{e}")
            entities = []

        return entities

    def raw_speak(self, image_path: str, description: str) -> str:
        """
        从图片和描述文本中提取实体
        :param image_path: 图片路径
        :param description: 图片描述文本
        :return: 提取的实体 JSON 数据
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": description,
                    },
                ],
            }
        ]

        # 准备推理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 推理生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0]
        return response


    def annotate_and_save_image(self, image_path: str, entities: dict, output_path: str):
        """
        标注实体框并保存图片
        :param image_path: 图片路径
        :param entities: 提取的实体 JSON 数据
        :param output_path: 保存标注后的图片路径
        """
        if not entities:
            logger.warning("没有实体数据可供标注")
            return

        # 读取原图
        image = cv2.imread(image_path)

        for entity in [entities]:
            name = entity["name"]
            bbox = entity["bnd"]

            # 提取边界框坐标
            left, top, right, bottom = map(int, bbox)
            # 在图片上绘制矩形框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # 添加文本标签
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存绘制后的图片
        cv2.imwrite(output_path, image)
        logger.info(f"绘制后的图片已保存到：{output_path}")

    def reflect_number(self, image_path: str, text: str, entity, bnd, description) -> str:
        description = f"Please analyze how many {entity} there are in the diagram, and whether there are any omissions in the following marks: {bnd} If there is an omission, please return 0, not 1. You don't need to return any explanation, just a number.{description}"
        ans = self.raw_speak(image_path, description)
        return int(ans)
    def reflect_correct(self, text: str, image_path: str, entity, description) -> str:
        bnd = entity["bnd"]
        name = entity["name"]
        if bnd == "null":
            return 1
        prompt = f"{text} Please look at the whole picture, analyze whether the entity in {bnd} is {name}. Note that the pictures from the interior of the aircraft cannot be regarded as this aircraft. Correctly returns 1, error returns 0. And briefly Explain why. You should return in json format: {{'reason': 'explanation', 'correct': 1 or 0}}"

        ans = self.raw_speak(image_path, prompt)
        # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
        start = ans.find("{")
        end = ans.find("}") + 1
        json_str = ans[start:end]

        # Step 2: 正常解析 JSON
        result = json.loads(json_str)

        return int(result['correct'])

    # def generate_prompt(self, text: str, entity_text: str, entity_label: str, description: str) -> str:
    #     prompt = (
    #         "Your task is to locate the specified entity in the image based on the provided description.\n"
    #         "\nPlease perform the following steps:\n"
    #         "1. Identify all instances of the entity in the image\n"
    #         "2. Return the bounding box coordinates [Xmin, Ymin, Xmax, Ymax] for each instance\n"
    #         "3. Output results in JSON format as shown in the examples below\n"
    #         "\nOutput format requirements:\n"
    #         "- If entity is found:\n"
    #         "   [{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}]\n"
    #         "- If entity is not found:\n"
    #         "   [{'name': 'Entity Name', 'bnd': 'null'}]\n"
    #         "If the entity is found:\n"
    #         "[{'name': 'fire_hydrant', 'bnd': [150, 300, 200, 350]}]\n"
    #         "If multiple instances are found:\n"
    #         "[{'name': 'person', 'bnd': [50, 100, 100, 200]}, {'name': 'person', 'bnd': [150, 120, 200, 220]}]\n"
    #         "If entity is not present:\n"
    #         "[{'name': 'fire_hydrant', 'bnd': 'null'}]\n"
    #         "**Begin your task:**"
    #         f"{text}.\n"
    #         f"Entity to locate: '{entity_text}' (Type: {entity_label}).\n"
    #         f"Description: {description}\n"
    #     )
    #     return prompt

    def generate_prompt(self, text: str, entity_text: str, entity_label: str, description: str, pre_judge_num='') -> str:
        buchong = ''
        if entity_label == "aircraft":
            buchong = 'Make sure to select the wings of the helicopter completely'
        if entity_label == "vehicle":
            buchong = 'Note that all the wheels of the vehicle should be framed completely.'
        if pre_judge_num=='':
            prompt = f"{text}\nPlease locate the position of {entity_text} in the image which is a {entity_label}. {description} If the entity appears multiple times, capture all instances. Output in JSON format: [{{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}}, {{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}}], ensuring the coordinates are accurate. If the entity is not appear in the image, then output in JSON format: [{{'name': 'Entity Name', 'bnd': 'null'}}]. If you believe the entity is not present in the image, you can confidently return `null`."
        else :
            prompt = f"{text}\nPlease locate the position of {entity_text} in the image which is {entity_label}. {description} For those covered by something, it is necessary to imagine the complete entity beneath the shelter and frame it completely.{buchong} If the entity appears multiple times, capture all instances. Output in JSON format: [{{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}}, {{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}}], ensuring the coordinates are accurate and completely."
        return prompt
#  It may have appeared {pre_judge_num} times in the image, and it is also possible that it will appear more times.

    def pre_judge(self, image_path, text: str, entity_text: str, entity_label: str, description: str) -> str:
        prompt = f"Please count the number of {entity_text} in the image, please find as many as possible. \n{text}\n{description} \nNote that the pictures from the interior of the aircraft cannot be regarded as this aircraft.\nIf it appears, count the number of instances and return the count as a number. You need to return as many possible items as possible. If you are very sure that it does not appear in the image, return 0. You don't need to give any explanation, just return a number."
        ans = self.raw_speak(image_path, prompt)
        return int(ans)

# ========== 主程序 ==========
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = './models/qwen'

    # 创建图像实体提取器实例
    extractor = ImageEntityExtractor(MODEL_PATH)

    # 图片路径和描述文本
    image_path = "./数据集/sample_image/0.jpg"
    description = "Soldiers test the overall system demonstrator armoured infantry fighting vehicle PUMA.请从这个句子中提取出相应的实体，并从图中找到对应的实体位置。以JSON格式输出：[{'name': '实体名称', 'bbox_2d': [左上x, 左上y, 右下x, 右下y]}]，确保坐标准确。"

    # 提取实体
    entities = extractor.extract_entities_from_image(image_path, description)
    print(f"Extracted entities: {entities}")

    # 标注实体框并保存图片
    output_path = "./output_image.jpg"
    extractor.annotate_and_save_image(image_path, entities, output_path)