import json
import re
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = './qwen'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./src/数据集/sample_image/0.jpg",  # 替换为你的图片路径
            },
            {
                "type": "text",
                "text": "Soldiers test the overall system demonstrator armoured infantry fighting vehicle PUMA.请从这个句子中提取出相应的实体，并从图中找到对应的实体位置。以JSON格式输出：[{'name': '实体名称', 'bbox_2d': [左上x, 左上y, 右下x, 右下y]}]，确保坐标准确。",
            },
        ],
    }
]

# 准备推理输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 推理生成输出
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 打印生成的文本
print("生成的文本：", output_text)

response = output_text[0]

try:
    # Step 1: 优先提取第一个完整的 JSON 数组（从 [ 开始到 ] 结束）
    start = response.find("[")
    end = response.rfind("]") + 1
    json_str = response[start:end]

    # Step 2: 正常解析 JSON
    entities = json.loads(json_str)

except Exception as e:
    logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
    entities = []

result_json = entities

# 如果需要进一步处理JSON数据（例如绘制矩形框），可以在下面添加代码
if result_json:
    # 示例：处理JSON数据
    for entity in result_json:
        name = entity.get("name")
        bbox = entity.get("bbox_2d")
        print(f"实体名称：{name}，边界框坐标：{bbox}")
else:
    print("无法提取JSON数据")


if result_json:
    # 读取原图
    image_path = "./src/数据集/sample_image/0.jpg"  # 替换为你的图片路径
    image = cv2.imread(image_path)

for entity in result_json:
    name = entity["name"]
    bbox = entity["bbox_2d"]

    # 提取边界框坐标
    left, top, right, bottom = map(int, bbox)

    # 在图片上绘制矩形框
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # 添加文本标签
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存绘制后的图片
output_path = "./output_image.jpg"
cv2.imwrite(output_path, image)
print(f"绘制后的图片已保存到：{output_path}")