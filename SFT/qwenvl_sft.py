import json
import os
from datasets import Dataset, load_from_disk
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
from Api_user import APIEntityExtractor
from PIL import Image
import numpy as np
import transformers
from torchvision import transforms
from transformers import AutoImageProcessor, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ========== APIEntityExtractor ==========
# 配置 OpenAI API 密钥、模型名称和基础网址
api_key = ""  # 替换为你的 OpenAI API 密钥
model_name = "gpt-3.5-turbo"  # 或者使用 "gpt-4" 等其他模型
api_base = "https://xh.v1api.cc/v1/"  # 替换为你的 OpenAI API 基础网址
default_headers = {"x-foo": "true"}  # 替换为你的默认头信息
# 创建实体提取器实例
APIextractor = APIEntityExtractor(api_key, model_name, api_base, default_headers)


class DataPreprocessor:
    def __init__(self, data_path, ans_path, image_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(ans_path, "r", encoding="utf-8") as f:
            self.ans = json.load(f)
        self.image_path = image_path

    def process_data(self):
        processed_data = []
        for idx, item in self.texts.items():
            text = item["text"]
            entities = self.ans.get(idx, [])
            if not entities:
                continue  # 跳过没有实体的样本

            formatted_entities = [{"name": entity["name"], "label": entity["label"], "bnd": entity["bnd"]} for entity in
                                  entities]
            response_json = json.dumps(formatted_entities, ensure_ascii=False)

            prompt = self._generate_prompt(text, entities)
            if prompt is None:
                continue

            # 加载和处理图片
            image_name = f"{idx}.jpg"
            image_path = os.path.join(self.image_path, image_name)
            if not os.path.exists(image_path):
                print(f"找不到图片文件：{image_path}，跳过此样本。")
                continue

            processed_data.append({
                "instruction": prompt,
                "completion": response_json,
                "image_path": image_path  # 保存路径
            })

        return processed_data

    def _generate_prompt(self, text, entities):
        entity_text = entities[0]["name"] if entities else "未知实体"
        entity_label = entities[0]["label"] if entities else "未知类别"
        if entity_text == "未知实体":
            return None
        description = APIextractor.get_description_from_openai(text=text, entity_name=entity_text)
        prompt = f"{text}.\nPlease locate the position of {entity_text} in the image which is a {entity_label}. {description} If the entity appears multiple times, capture all instances. Output in JSON format: [{{'name': 'Entity Name', 'bnd': [Xmin, Ymin, Xmax, Ymax]}}], ensuring the coordinates are accurate. If the entity is not appear in the image, then output in JSON format: [{{'name': 'Entity Name', 'bnd': 'null'}}]. If you believe the entity is not present in the image, you can confidently return `null`. For example, entities like 'location' are likely not present, so do not hesitate to use `null`."
        return prompt


def load_and_prepare_data(train_text_path, train_entity_path, val_text_path, val_entity_path, image_path):
    # 加载和准备数据
    train_preprocessor = DataPreprocessor(train_text_path, train_entity_path, image_path)
    train_data = train_preprocessor.process_data()
    train_dataset = Dataset.from_list(train_data)

    val_preprocessor = DataPreprocessor(val_text_path, val_entity_path, image_path)
    val_data = val_preprocessor.process_data()
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def main():
    # 数据路径
    train_text_path = "./数据集/train_text.json"
    train_entity_path = "./数据集/train_entity.json"
    val_text_path = "./数据集/val_text.json"
    val_entity_path = "./数据集/val_entity.json"
    image_path = "./数据集/sample_image"  # 图片文件夹路径

    # 加载和准备数据
    # train_dataset, val_dataset = load_and_prepare_data(
    #     train_text_path, train_entity_path, val_text_path, val_entity_path, image_path
    # )

    # train_dataset.save_to_disk(os.path.join("./数据集", "train_dataset"))
    # val_dataset.save_to_disk(os.path.join("./数据集", "val_dataset"))
    # print(f"训练数据集已保存至：train_dataset")
    # print(f"验证数据集已保存至：val_dataset")

    # 加载保存的数据集
    train_dataset = load_from_disk("./数据集/train_dataset")
    val_dataset = load_from_disk("./数据集/val_dataset")

    # 加载模型和分词器
    model_path = "./models/qwen"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 定义数据处理函数
    def data_collator(features):
        # 提取所有文本和图像路径
        texts = []
        image_paths = []
        for feature in features:
            full_text = feature['instruction'] + feature['completion']
            texts.append(full_text)
            image_paths.append(feature["image_path"])

        # 将图像路径转换为 Image 对象
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            images.append(image)

        # 使用模型内置 processor 统一处理文本和图像
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # 处理 labels
        labels = []
        for i, text in enumerate(texts):
            input_ids = inputs.input_ids[i]
            # 获取 instruction 的 token 长度
            instruction = features[i]['instruction']
            instruction_tokenized = tokenizer(instruction, return_tensors="pt", max_length=1024, truncation=True)
            instruction_len = instruction_tokenized.input_ids.shape[1]

            label_ids = input_ids.clone()
            label_ids[:instruction_len] = -100
            labels.append(label_ids)

        # 填充 labels
        max_label_length = max(len(label) for label in labels)
        padded_labels = torch.full((len(labels), max_label_length), -100, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label

        # 动态计算 image_grid_thw
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = None
        if "pixel_values" in inputs:
            pixel_values = inputs.pixel_values
            # 检查 pixel_values 的形状
            if len(pixel_values.shape) == 4:  # [B, C, H, W]
                _, _, h, w = pixel_values.shape
                image_grid_thw = [(1, h // 16, w // 16) for _ in range(pixel_values.size(0))]
            elif len(pixel_values.shape) == 3:  # [C, H, W]（无批次维度）
                _, h, w = pixel_values.shape
                image_grid_thw = [(1, h // 16, w // 16)] * len(texts)  # 假设批次大小等于文本数量
            else:
                # print("Unexpected pixel_values shape:", pixel_values.shape)
                pass

        return {
            "input_ids": inputs.input_ids,
            "labels": padded_labels,
            "attention_mask": inputs.attention_mask,
            # "pixel_values": inputs.get("pixel_values", None),
            "image_grid_thw": image_grid_thw
        }

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./output/qwen_lora_sft",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./output/qwen_lora_sft/logs",
        logging_steps=1,
        learning_rate=3e-4,
        save_total_limit=1000,
        weight_decay=0.01,
        # fp16=True,
        bf16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 开始微调
    trainer.train()

    # 保存微调后的模型
    model.save_pretrained("./output/qwen_lora_sft/merged_model")
    tokenizer.save_pretrained("./output/qwen_lora_sft/merged_model")


if __name__ == "__main__":
    main()