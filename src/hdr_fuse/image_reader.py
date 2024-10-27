# src/image_reader.py

"""
图像读取组件，使用Pillow库读取图像文件并转换为Pillow Image对象。
"""

import logging
from PIL import Image
from typing import List
from .exceptions import ImageReadError

logger = logging.getLogger(__name__)

class ImageReader:
    def __init__(self):
        logger.debug("初始化 ImageReader 组件。")

    def read_images(self, paths: List[str]) -> List[Image.Image]:
        """
        读取多个图像文件，返回Pillow的Image对象列表。

        :param paths: 图像文件路径的字符串列表
        :return: Pillow的Image对象列表
        """
        logger.info(f"开始读取 {len(paths)} 张图像。")
        images = []
        for path in paths:
            try:
                logger.debug(f"读取图像: {path}")
                img = Image.open(path).convert('RGB')  # 确保图像为RGB格式
                images.append(img)
                logger.debug(f"成功读取图像: {path}，大小: {img.size}")
            except Exception as e:
                logger.error(f"读取图像失败: {path}，错误: {e}")
                raise ImageReadError(f"Failed to read image at {path}: {e}")
        logger.info(f"成功读取了 {len(images)} 张图像。")
        return images
