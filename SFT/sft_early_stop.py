import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset  # This is from Hugging Face's datasets library
from sklearn.model_selection import train_test_split

# ========== 环境配置 ==========

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "./models/glm3-6b"
local_cache_dir = "./models/glm3-6b"
output_dir = "./output/sft_lora_glm3"  # Directory to save fine-tuned model and checkpoints

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ========== 加载模型和分词器 ==========
print(f"🔄 正在加载 {model_name} 模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=local_cache_dir,
    local_cache_dir=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=local_cache_dir,
    local_files_only=True,
    torch_dtype=torch.float16,  # Use float16 for reduced memory usage
).to(device)
model.eval()  # Set to eval for initial loading, will switch to train later
print("✅ 模型和分词器加载完成")

# 清理显存
torch.cuda.empty_cache()

# ========== 配置 LoRA ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA attention dimension
    lora_alpha=32,  # The alpha parameter for LoRA scaling
    lora_dropout=0.1,  # Dropout probability for LoRA layers
    target_modules=["query_key_value"],  # Modules to apply LoRA to
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ LoRA 配置完成，模型已准备好进行微调")


# ========== 数据准备函数 ==========
def prepare_data_for_sft(sample_text_path: str, sample_entity_path: str):
    print("🔄 正在准备训练数据...")
    with open(sample_text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(sample_entity_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    processed_data = []
    for idx, text_item in tqdm(texts.items(), desc="处理数据"):
        text = text_item["text"]
        entity_list = labels.get(idx, [])

        formatted_entities = []
        for entity in entity_list:
            formatted_entity = {}
            if "name" in entity:
                formatted_entity["name"] = entity["name"]
            if "label" in entity:
                formatted_entity["label"] = entity["label"]

            if formatted_entity:
                formatted_entities.append(formatted_entity)

        response_json = json.dumps(formatted_entities, ensure_ascii=False, separators=(',', ':'))

        prompt = f"""
        You are an expert in military information extraction. Your job is to extract Military-related named entities from military-related English text, and classify each entity into one of the following 6 types:
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
        3. Use the exact JSON array format.
        4. Do not explain or describe the output. Just return pure JSON.

        --- Examples ---

        Text: "The F-16 Fighting Falcon is an American fighter aircraft. It was deployed in Afghanistan."
        Output:
        [
            {{"name":"F-16","label":"aircraft"}}
        ]

        --- Now extract entities from the following text ---

        Text: "{text}"
        Output:
        """
        processed_data.append({
            "instruction": prompt,
            "completion": response_json
        })
    print("✅ 数据准备完成")
    return processed_data


# ========== 数据加载和分割 ==========
sample_text_path = "./数据集/sample_text_filter.json"
sample_entity_path = "./数据集/sample_entity.json"

if not os.path.exists(sample_text_path):
    print(f"❌ 找不到输入文件：{sample_text_path}")
    exit()
if not os.path.exists(sample_entity_path):
    print(f"❌ 找不到标签文件：{sample_entity_path}")
    exit()

raw_data = prepare_data_for_sft(sample_text_path, sample_entity_path)

if not raw_data:
    print("⚠️ 原始数据为空，无法进行训练和验证集的分割。请检查数据集文件。")
    exit()

train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")


# ========== 数据 collator 函数 ==========
def data_collator(features):
    tokenized_inputs = []
    labels = []

    for feature in features:
        full_text = f"{feature['instruction']}{feature['completion']}"
        tokenized = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = tokenized.input_ids[0]

        instruction_tokenized = tokenizer(feature['instruction'], return_tensors="pt", max_length=1024, truncation=True)
        instruction_len = instruction_tokenized.input_ids.shape[1]

        label_ids = input_ids.clone()
        label_ids[:instruction_len] = -100

        tokenized_inputs.append(input_ids)
        labels.append(label_ids)

    max_len = max(len(t) for t in tokenized_inputs)
    padded_input_ids = torch.full((len(tokenized_inputs), max_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)

    for i, (input_id_tensor, label_id_tensor) in enumerate(zip(tokenized_inputs, labels)):
        padded_input_ids[i, :len(input_id_tensor)] = input_id_tensor
        padded_labels[i, :len(label_id_tensor)] = label_id_tensor

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": (padded_input_ids != tokenizer.pad_token_id).long()
    }


# ========== 自定义 EarlyStopping Callback ==========
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.wait = 0


def on_evaluate(self, args, state, control, metrics, **kwargs):
    # 获取验证集损失
    eval_loss = metrics.get('eval_loss')
    print('哈哈哈')
    # 检查是否有改进
    if eval_loss < self.best_loss:
        self.best_loss = eval_loss
        self.wait = 0

        # 保存验证效果最好的模型
        best_model_path = os.path.join(args.output_dir, "best_model")
        self.model.save_pretrained(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        print(f"\n✅ 验证效果最好的模型已保存至：{best_model_path}")
    else:
        self.wait += 1

    # 如果等待次数超过耐心值，触发早停
    if self.wait >= self.patience:
        control.should_training_stop = True
        print(f"\n✅ Early stopping triggered after {self.patience} epochs without improvement.")

    return control

    def generate_and_save_validation_results(self, args, state, control, metrics):
        # 验证集生成结果
        validation_results = []
        for example in val_dataset:
            prompt = example['instruction']
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                try:
                    # 这里修改了 generate 的调用方式，明确指定参数名称
                    output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=256,
                        do_sample=False,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=256,
                        do_sample=False,
                    )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            output_start_str = "Output:\n"
            output_start_index = response.rfind(output_start_str)
            if output_start_index != -1:
                generated_text = response[output_start_index + len(output_start_str):].strip()
            else:
                generated_text = response.strip()

            try:
                start = generated_text.find("[")
                end = generated_text.rfind("]") + 1
                json_str = generated_text[start:end]
                entities = json.loads(json_str)
                validation_results.append({
                    "input": prompt,
                    "output": entities
                })
            except Exception as e:
                print(f"⚠️ JSON 解析失败: {generated_text}")

        # 保存验证集生成结果
        validation_results_path = os.path.join(output_dir, "validation_results.json")
        with open(validation_results_path, "w", encoding="utf-8") as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        print(f"✅ 验证集生成结果已保存至：{validation_results_path}")


# ========== 训练参数配置 ==========
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=100,  # 增加到100个epoch
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",  # 在每个epoch结束时进行验证
    save_strategy="steps",
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_steps=1,
    learning_rate=3e-4,
    weight_decay=0.01,
    fp16=False,  # 禁用 FP16 混合精度训练
    bf16=True,  # 使用 BF16（如果 GPU 支持）
    optim="paged_adamw_8bit",
    load_best_model_at_end=True,  # 加载验证集上表现最好的模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
)

# ========== 训练器初始化和启动 ==========
print("🚀 正在初始化训练器...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(patience=3, threshold=0.001)]  # 添加早停回调
)

print("⚡️ 开始模型训练...")
trainer.train()
print("✅ 模型训练完成")

# ========== 保存微调后的 LoRA 适配器 ==========
lora_adapter_path = f"{output_dir}/lora_adapters"
trainer.save_model(lora_adapter_path)
print(f"✅ LoRA 适配器已保存至：{lora_adapter_path}")

try:
    print("🔄 正在合并 LoRA 适配器到基础模型...")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=local_cache_dir,
        torch_dtype=torch.float16
    ).to(device)

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model_path = f"{output_dir}/merged_model"
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"✅ 合并后的模型已保存至：{merged_model_path}")

except Exception as e:
    print(f"⚠️ 无法合并 LoRA 适配器到基础模型：{e}")
    print("您仍然可以使用单独的 LoRA 适配器和原始模型进行推理。")

# ========== 推理示例 (使用微调后的LoRA适配器) ==========
print("\n--- 推理示例 ---")


def infer_with_lora(text: str, peft_model_path: str, base_model_path: str, cache_dir: str) -> list:
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch.float16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    model_with_lora = PeftModel.from_pretrained(base_model, peft_model_path)
    model_with_lora.eval()

    prompt = f"""
    You are an expert in military information extraction. Your job is to extract Military-related named entities from military-related English text, and classify each entity into one of the following 6 types:
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
    3. Use the exact JSON array format.
    4. Do not explain or describe the output. Just return pure JSON.

    --- Examples ---

    Text: "The F-16 Fighting Falcon is an American fighter aircraft. It was deployed in Afghanistan."
    Output:
    [
        {{"name":"F-16","label":"aircraft"}}
    ]

    --- Now extract entities from the following text ---

    Text: "{text}"
    Output:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        try:
            output = model_with_lora.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            output = model_with_lora.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
            )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    output_start_str = "Output:\n"
    output_start_index = response.rfind(output_start_str)
    if output_start_index != -1:
        generated_text = response[output_start_index + len(output_start_str):].strip()
    else:
        generated_text = response.strip()

    try:
        start = generated_text.find("[")
        end = generated_text.rfind("]") + 1
        json_str = generated_text[start:end]
        entities = json.loads(json_str)
        return entities
    except Exception as e:
        print(f"⚠️ JSON 解析失败: {e}")
        return []