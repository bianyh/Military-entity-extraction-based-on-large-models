import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 环境配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "./models/chatglm3-6b"
local_cache_dir = "./models/chatglm3-6b"

# ========== 加载模型 ==========
print(f"🔄 正在加载 {model_name} 模型...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=local_cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=local_cache_dir
).half().to(device)
model.eval()
print("✅ 模型加载完成")

# ========== 实体抽取函数 ==========
def query_chatglm3(text: str) -> list:
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

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.3,
            top_p=0.7
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt)+9:]
    print(f"\n🔍 [DEBUG] 模型原始输出:\n{response.strip()}\n")

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

    except:
        print("⚠️ json.loads 失败，返回空列表")
        return []

# ========== 主程序 ==========
def main():
    input_path = "./数据集/sample_text_filter.json"
    output_path = "./output/entity_text_only_original.json"

    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件：{input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    results = {}

    for idx, item in tqdm(texts.items(), desc="📍 正在抽取实体"):
        text = item["text"]
        try:
            entities = query_chatglm3(text)
            results[idx] = entities
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



# ChatGlm3-6B:
# Precision: 0.1442
# Recall: 0.5222
# F1-Score: 0.2260

# qwen3-14B:
# Precision: 0.4000
# Recall: 0.3778
# F1-Score: 0.3886