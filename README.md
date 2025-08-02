# 基于大模型的军事实体提取

## 环境要求

- **Python**: 3.10
- **PyTorch**: 2.6.0
- **CUDA**: 11.8

## 模型准备

请确保模型托管平台的下载权限已配置正常，然后使用以下命令将所需模型下载到项目目录：

```bash
# 下载 GLM4-9B 模型到指定目录
modelscope download --model ZhipuAI/glm-4-9b-chat-hf --local_dir ./models/glm4-9b

# 下载 Qwen2.5-VL-32B 模型到指定目录
modelscope download --model Qwen/Qwen2.5-VL-32B-Instruct --local_dir ./models/qwen-vl-32b
```

如遇下载失败，请根据[ModelScope](https://www.modelscope.cn)平台说明调整下载策略。

## 环境配置

### 安装依赖

运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

### 配置 API 密钥

在项目根目录创建或编辑 `config.json` 文件，配置 API 密钥：

```json
{
  "api_key": "your_api_key_here",
  // 其他配置项...
}
```

## 系统架构

本项目采用模块化设计，主要包含以下系统模块：

### 流程系统

- **基础流程系统**：`foundation_workflow.py` 
- **反思流程系统**：
  - `reflect_workflow1.py` - 第一种简单反思优化流程
  - `reflect_workflow2.py` - 第二种复杂反思优化流程
  - `reflect_workflow3.py` - 长流程反思系统（第三种流程）

### 实体提取工具

- **文本实体提取**：`EntityExtractor.py` - 从文本内容中提取实体信息
- **图像实体提取**：`ImageEntityExtractor.py` - 从图像内容中提取实体信息

## 模型微调

微调生成的模型适配器存储路径：`./models/glm4-9b-lora` 

## 使用指南

根据实际需求选择运行对应的系统模块：

```bash
# 运行基础流程系统
python foundation_workflow.py

# 运行简单反思流程系统
python reflect_workflow1.py

# 运行复杂反思流程系统
python reflect_workflow2.py

# 运行长流程系统（反思流程3）
python reflect_workflow3.py
```

## 注意事项

1. 模型下载可能需要较长时间，建议在稳定网络环境下进行
2. 如遇 CUDA 兼容性问题，请参考[PyTorch 官方文档](https://pytorch.org/get-started/previous-versions/)调整配置
3. API 密钥需保持机密，避免泄露导致安全风险

## 技术支持

如遇代码运行问题，请联系我。