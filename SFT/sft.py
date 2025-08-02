import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset  # This is from Hugging Face's datasets library
from sklearn.model_selection import train_test_split

# ========== 环境配置 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "./models/glm4-9b"
local_cache_dir = "./models/glm4-9b"
output_dir = "./output/sft_lora_chatglm4"  # Directory to save fine-tuned model and checkpoints

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ========== 加载模型和分词器 ==========
print(f"正在加载 {model_name} 模型和分词器...")
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
model.train()  # Set to eval for initial loading, will switch to train later
print("模型和分词器加载完成")

# 清理显存
torch.cuda.empty_cache()

# # ========== 加载模型和分词器 ==========
# print(f"正在加载 {model_name} 模型和分词器...")
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     cache_dir=local_cache_dir,
#     local_cache_dir=True
# )
# # ========== 模型加载部分（修改） ==========
#
# # ========== 模型加载部分（修改） ==========
# quantization_config = BitsAndBytesConfig(
# load_in_4bit=True,
# bnb_4bit_compute_dtype=torch.float16,
# bnb_4bit_quant_type="nf4",
# bnb_4bit_use_double_weight=True
# )
# model = AutoModelForCausalLM.from_pretrained(
# model_name,
# trust_remote_code=True,
# cache_dir=local_cache_dir,
# local_files_only=True,
# torch_dtype=torch.float16,  # 使用 float16 减少显存占用
# device_map="auto",  # 自动分配模型到多个 GPU（如果有多个 GPU）
# quantization_config=quantization_config  # 应用量化配置
# ).to(device)
# model.train()
# print("模型和分词器加载完成")
#
# # 清理显存
# torch.cuda.empty_cache()

# ========== 配置 LoRA ==========
# Define LoRA configuration
# target_modules should be chosen based on the model's architecture.
# For ChatGLM3, 'query_key_value' is a common choice for attention layers.
# You might also include 'dense' or other linear layers depending on experimentation.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA attention dimension
    lora_alpha=32,  # The alpha parameter for LoRA scaling
    lora_dropout=0.1,  # Dropout probability for LoRA layers
    target_modules=['q_proj', 'k_proj', 'v_proj'],  # Modules to apply LoRA to
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("LoRA 配置完成，模型已准备好进行微调")


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


# ========== 数据准备函数 ==========
def prepare_data_for_sft(sample_text_path: str, sample_entity_path: str):
    """
    Loads text and entity data, combines them, and formats for SFT.
    """
    print("正在准备训练数据...")
    with open(sample_text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(sample_entity_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    processed_data = []
    for idx, text_item in tqdm(texts.items(), desc="处理数据"):
        text = text_item["text"]
        entity_list = labels.get(idx, [])  # Get entities for the current ID, default to empty list if not found

        # Format entities as a JSON string for the model to generate
        # Remove 'bnd' key as it's not part of the desired output format
        formatted_entities = []
        for entity in entity_list:
            # Ensure only 'name' and 'label' are included, and handle potential missing keys gracefully
            formatted_entity = {}
            if "name" in entity:
                formatted_entity["name"] = entity["name"]
            if "label" in entity:
                formatted_entity["label"] = entity["label"]

            if formatted_entity:  # Only add if it's not empty after filtering
                formatted_entities.append(formatted_entity)

        formatted_entities = remove_duplicate_entities(formatted_entities)

        # Ensure the output is a compact JSON string without extra whitespace for consistency
        response_json = json.dumps(formatted_entities, ensure_ascii=False, separators=(',', ':'))

        # Construct the instruction-response pair as expected by the model
        # This prompt should match the inference prompt from your original code
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
        # The model should learn to generate just the JSON output
        # For SFT, the 'completion' is what the model should generate
        processed_data.append({
            "instruction": prompt,
            "completion": response_json
        })
    print("数据准备完成")
    return processed_data


# ========== 数据加载和分割 ==========
# sample_text_path = "./数据集/train_text.json"
# sample_entity_path = "./数据集/train_entity.json"  # Your labeled data

# if not os.path.exists(sample_text_path):
#     print(f"找不到输入文件：{sample_text_path}")
#     exit()
# if not os.path.exists(sample_entity_path):
#     print(f"找不到标签文件：{sample_entity_path}")
#     exit()

# raw_data = prepare_data_for_sft(sample_text_path, sample_entity_path)

# # Split data into training and validation sets
# # Added a check to ensure raw_data is not empty before splitting
# if not raw_data:
#     print("原始数据为空，无法进行训练和验证集的分割。请检查数据集文件。")
#     exit()

# train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)  # 10% for validation

# # Convert to Hugging Face Dataset format
# train_dataset = Dataset.from_list(train_data)
# val_dataset = Dataset.from_list(val_data)

# print(f"训练集大小: {len(train_dataset)}")
# print(f"验证集大小: {len(val_dataset)}")

# ========== 数据加载和分割 修改部分 ==========
sample_text_path = "./数据集/sample_text.json"
sample_entity_path = "./数据集/sample_entity.json"  # Your labeled data
val_text_path = "./数据集/val_text.json"
val_entity_path = "./数据集/val_entity.json"  # Your validation labeled data

# Check if all necessary files exist
if not os.path.exists(sample_text_path):
    print(f"找不到输入文件：{sample_text_path}")
    exit()
if not os.path.exists(sample_entity_path):
    print(f"找不到标签文件：{sample_entity_path}")
    exit()
if not os.path.exists(val_text_path):
    print(f"找不到验证输入文件：{val_text_path}")
    exit()
if not os.path.exists(val_entity_path):
    print(f"找不到验证标签文件：{val_entity_path}")
    exit()

# Prepare training and validation data separately
train_data = prepare_data_for_sft(sample_text_path, sample_entity_path)
val_data = prepare_data_for_sft(val_text_path, val_entity_path)

# Check if data is empty
if not train_data:
    print("训练数据为空，无法进行训练。请检查数据集文件。")
    exit()
if not val_data:
    print("验证数据为空，无法进行验证。请检查验证集文件。")
    exit()

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")


# ========== 数据 collator 函数 ==========
def data_collator(features):
    """
    A simple data collator that tokenizes and formats the input for the model.
    """
    tokenized_inputs = []
    labels = []

    for feature in features:
        # For ChatGLM-like models, it's common to concatenate instruction and completion
        # and then set labels for completion part only (or ignore instruction loss).
        # The tokenizer's `apply_chat_template` or a similar method might be used for chat models.
        # For simplicity here, we concatenate directly for SFT.

        # Construct the full text the model should see (input + target output)
        full_text = f"{feature['instruction']}{feature['completion']}"

        # Tokenize the combined text
        # Ensure max_length is sufficient for your longest combined texts
        tokenized = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)  # Increased max_length
        input_ids = tokenized.input_ids[0]

        # Tokenize only the instruction to determine its length
        instruction_tokenized = tokenizer(feature['instruction'], return_tensors="pt", max_length=1024, truncation=True)
        instruction_len = instruction_tokenized.input_ids.shape[1]

        # Create labels: -100 for instruction tokens, actual token IDs for completion tokens
        # -100 is the default ignore_index for PyTorch's CrossEntropyLoss
        label_ids = input_ids.clone()
        label_ids[:instruction_len] = -100  # Mask out the instruction part

        tokenized_inputs.append(input_ids)
        labels.append(label_ids)

    # Pad inputs and labels to the longest sequence in the batch
    # Pad input_ids with tokenizer.pad_token_id
    # Pad labels with -100
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


# ========== 训练参数配置 ==========
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size per GPU for training
    per_device_eval_batch_size=1,  # Batch size per GPU for evaluation
    gradient_accumulation_steps=1,  # Accumulate gradients over 8 steps to simulate a larger batch size
    eval_strategy="epoch",  # Evaluate every `eval_steps`
    eval_steps=1,  # Evaluation interval
    save_strategy="epoch",  # Save checkpoint every `save_steps`
    save_steps=1,  # Save checkpoint interval
    save_total_limit=100000,  # Only keep the last 2 checkpoints
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,  # Log training metrics every 10 steps
    learning_rate=3e-4,  # Learning rate for LoRA
    weight_decay=0.01,  # Weight decay
    # fp16=True,  # Use mixed precision training if available
    bf16=True,  # Use bfloat16 if your GPU supports it (e.g., Ampere architecture)
    optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer for memory efficiency
    report_to="none",  # Do not report to any platform (e.g., wandb)
    load_best_model_at_end=True,  # Load the best model found during training
    metric_for_best_model="eval_loss",  # Metric to use for loading the best model
    greater_is_better=False,  # Lower eval_loss is better
    remove_unused_columns=False,  # ⚡️ Crucial fix: Prevent Trainer from removing columns ⚡️
)

# ========== 训练器初始化和启动 ==========
print("正在初始化训练器...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,  # Pass tokenizer to trainer
    data_collator=data_collator,
)

print("开始模型训练...")
trainer.train()
print("模型训练完成")

# ========== 保存微调后的 LoRA 适配器 ==========
# Save the LoRA adapters
lora_adapter_path = f"{output_dir}/lora_adapters"
trainer.save_model(lora_adapter_path)
print(f"LoRA 适配器已保存至：{lora_adapter_path}")

# If you want to merge the LoRA adapters with the base model and save the full model:
# This requires enough GPU memory to load the full base model and then merge.
try:
    print("正在合并 LoRA 适配器到基础模型...")
    # It's good practice to clear the current model from memory
    # before loading the base model again for merging.
    del model
    torch.cuda.empty_cache()  # Clear CUDA cache

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=local_cache_dir,
        torch_dtype=torch.float16
    ).to(device)

    from peft import PeftModel

    # Load the saved LoRA adapters onto the base model
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge LoRA weights into the base model
    merged_model = peft_model.merge_and_unload()
    merged_model_path = f"{output_dir}/merged_model"
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"合并后的模型已保存至：{merged_model_path}")

except Exception as e:
    print(f"无法合并 LoRA 适配器到基础模型 (可能是内存不足或 PEFT 版本问题)：{e}")
    print("您仍然可以使用单独的 LoRA 适配器和原始模型进行推理。")

# # ========== 推理示例 (使用微调后的LoRA适配器) ==========
# print("\n--- 推理示例 ---")


# def infer_with_lora(text: str, peft_model_path: str, base_model_path: str, cache_dir: str) -> list:
#     # 清理显存
#     torch.cuda.empty_cache()

#     # Load base model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         trust_remote_code=True,
#         cache_dir=cache_dir,
#         torch_dtype=torch.float16
#     ).to(device)

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_path,
#         trust_remote_code=True,
#         cache_dir=cache_dir
#     )

#     # Load PEFT model (LoRA adapters)
#     model_with_lora = PeftModel.from_pretrained(base_model, peft_model_path)
#     model_with_lora.eval()  # Set to evaluation mode

#     prompt = f"""You are an information extraction expert. Your job is to extract named entities from military-related English text, and classify each entity into one of the following types:
# ["vehicle", "aircraft", "vessel", "weapon", "location", "other"]

# Please follow these rules:
# 1. Only extract entities that explicitly appear in the text and only classify them in the 6 types.
# 3. Use the exact JSON array format.
# 3. Do not explain or describe the output. Just return pure JSON.

# --- Examples ---

# Text: "Soldiers test the overall system demonstrator armoured infantry fighting vehicle PUMA."
# Output:
# [
#   {{"name": "PUMA", "label": "vehicle"}}
# ]

# --- Now extract entities from the following text ---

# Text: "{text}"
# Output:
# """

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     with torch.no_grad():
#         try:
#             output = model_with_lora.generate(
#                 input_ids,
#                 max_new_tokens=256,
#                 do_sample=False,
#             )
#         except torch.cuda.OutOfMemoryError:
#             # 如果显存不足，清理显存并重试
#             torch.cuda.empty_cache()
#             output = model_with_lora.generate(
#                 input_ids,
#                 max_new_tokens=256,
#                 do_sample=False,
#             )

#     # Decode the output
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     # The response will contain the prompt itself, so we need to slice it
#     output_start_str = "Output:\n"
#     output_start_index = response.rfind(output_start_str)
#     if output_start_index != -1:
#         generated_text = response[output_start_index + len(output_start_str):].strip()
#     else:
#         generated_text = response.strip()  # Fallback if "Output:\n" not found

#     # Attempt to extract JSON
#     try:
#         start = generated_text.find("[")
#         end = generated_text.rfind("]") + 1
#         json_str = generated_text[start:end]
#         entities = json.loads(json_str)
#         return entities
#     except Exception as e:
#         print(f"JSON 解析失败: {e}")
#         return []


# # Example usage of the fine-tuned model
# if os.path.exists(lora_adapter_path):
#     test_text = "The F-16 Fighting Falcon is an American fighter aircraft. It was deployed in Afghanistan."
#     print(f"\n对文本进行推理: '{test_text}'")
#     extracted_entities = infer_with_lora(test_text, lora_adapter_path, model_name, local_cache_dir)
#     print(f"提取到的实体: {extracted_entities}")
# else:
#     print("\nLoRA 适配器未保存或不存在，跳过推理示例。")