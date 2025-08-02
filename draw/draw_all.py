import json
import os

import cv2
from loguru import logger


def annotate_and_save_image(image_path: str, entities: list[dict], output_path: str):
    """
    标注实体框并保存图片
    :param image_path: 图片路径
    :param entities: 提取的实体 JSON 数据列表
    :param output_path: 保存标注后的图片路径
    """
    if not entities:
        logger.warning("没有实体数据可供标注")
        return

    # 读取原图（只读取一次）
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"无法加载图像：{image_path}")
        return

    # 检查输出路径目录是否存在，不存在则创建
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录：{output_dir}")

    for entity in entities:  # 修复循环逻辑，直接遍历实体列表
        name = entity.get("name", "未知实体")
        bbox = entity.get("bnd", [])

        if not bbox or len(bbox) != 4:
            logger.warning(f"实体 '{name}' 的边界框数据无效：{bbox}")
            continue

        # 提取边界框坐标
        left, top, right, bottom = map(int, list(bbox.values()))
        # 在图片上绘制矩形框
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # 添加文本标签
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存绘制后的图片（所有实体绘制完成后才保存）
    cv2.imwrite(output_path, image)
    logger.info(f"绘制后的图片已保存到：{output_path}")


with open("./output/final_results.json", 'r') as f:
    data = json.load(f)
    for key, item in data.items():
        annotate_and_save_image(f'./dataset/test/test_image/{key}.jpg', item, f'./output/image/{key}.jpg')

