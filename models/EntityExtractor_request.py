import json
import requests
import os

class EntityExtractor:
    def __init__(self, input_path, output_path, api_url):
        """
        初始化 EntityExtractor 类
        :param input_path: 输入数据集文件路径
        :param output_path: 输出结果文件路径
        :param api_url: Flask 后端接口地址
        """
        self.input_path = input_path
        self.output_path = output_path
        self.api_url = api_url

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def read_dataset(self):
        """
        读取输入数据集文件
        :return: 数据集内容
        """
        if not os.path.exists(self.input_path):
            print(f"找不到输入文件：{self.input_path}")
            exit(1)

        with open(self.input_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
        return texts

    def send_request(self, text):
        """
        向 Flask 后端接口发送文本数据并获取结果
        :param text: 待处理的文本
        :return: 返回的实体列表
        """
        try:
            response = requests.post(self.api_url, json={"text": text})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"请求失败，状态码：{response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"请求异常：{e}")
            return []

    def process_texts(self):
        """
        处理数据集中的所有文本
        """
        texts = self.read_dataset()
        results = {}

        for idx, item in texts.items():
            text = item["text"]
            print(f"处理 ID: {idx}, 文本: {text}")

            entities = self.send_request(text)
            results[idx] = entities
            print(f"成功获取结果：{entities}")

        self.save_results(results)

    def save_results(self, results):
        """
        将结果保存到文件
        :param results: 处理结果
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"所有数据处理完成，结果已保存到：{self.output_path}")


# 使用示例
if __name__ == "__main__":
    input_path = "./数据集/sample_text_filter.json"  # 输入数据集文件
    output_path = "./output/results.json"  # 输出结果文件
    api_url = "http://localhost:5000/extract_entities"  # Flask 后端接口地址

    extractor = EntityExtractor(input_path, output_path, api_url)
    extractor.process_texts()