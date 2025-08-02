# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# model_name = "THUDM/chatglm3-6b"
# local_dir = "./models/chatglm3-6b"
#
# # 第一次联网加载并保存到本地
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.save_pretrained(local_dir)
#
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# model.save_pretrained(local_dir)


from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True
)

model.save_pretrained("./models/qwen2.5-vl-3B", safe_serialization=True)
processor.save_pretrained("./models/qwen2.5-vl-3B")

