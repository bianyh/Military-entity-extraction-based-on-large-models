import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger


# 推理示例函数
def inference_with_lora(text: str, max_new_tokens: int = 256) -> str:
    """
    使用微调后的 LoRA 模型进行推理
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = model.base_model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model.base_model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            return entities

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
            return entities

        except Exception as e:
            logger.warning(f"⚠️ json.loads 失败，返回空列表。错误信息：{e}")
            return []


# ========== 主程序 ==========
def main():
    input_path = "./数据集/val_text.json"
    # 构建输出路径
    output_dir = f"./output/{model_name}/{weitiao_banben}"
    output_path = os.path.join(output_dir, "result.json")  # 假设最终输出文件名是 result.json

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"❌ 找不到输入文件：{input_path}")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✅ 输出目录已准备就绪：{output_dir}")

    with open(input_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    results = {}

    for idx, item in tqdm(texts.items(), desc="📍 正在抽取实体"):
        text = item["text"]
        try:
            logger.debug(f'处理文本{text}')
            entities = inference_with_lora(text)
            results[idx] = entities
        except Exception as e:
            logger.error(f"[Error] ID: {idx}, Error: {e}")
            results[idx] = []

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ 实体抽取完成，结果已保存至：{output_path}")


# 配置日志
log_path = "./log"
os.makedirs(log_path, exist_ok=True)
logger.add(f"{log_path}/{os.path.basename(__file__)}.log", rotation="100 MB")  # 自动分割日志文件

model_name = "glm4-9b"
nums = [280, 560, 840, 1120, 1400, 1680, 1960, 2240, 2520, 2800]
for i in nums:
    try:
        weitiao_banben = f"checkpoint-{i}"

        # 配置路径
        base_model_path = f"./models/{model_name}"  # 基础模型路径
        lora_adapter_path = f"./output/sft_lora_glm4/{weitiao_banben}"  # LoRA 适配器保存路径
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载基础模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        # 加载 LoRA 适配器到基础模型上
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model.eval()  # 设置为评估模式

        if __name__ == "__main__":
            main()
    except Exception as e:
        del model
        del tokenizer
        torch.cuda.empty_cache()

        weitiao_banben = f"checkpoint-{i}"

        # 配置路径
        base_model_path = f"./models/{model_name}"  # 基础模型路径
        lora_adapter_path = f"./output/sft_lora_glm4/{weitiao_banben}"  # LoRA 适配器保存路径
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载基础模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        # 加载 LoRA 适配器到基础模型上
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model.eval()  # 设置为评估模式

        if __name__ == "__main__":
            main()


