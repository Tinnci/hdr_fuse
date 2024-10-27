# src/hdr_fuse/image_writer.py

"""
图像输出组件，使用Pillow保存处理后的图像文件。

"""

import logging
from PIL import Image
from typing import List
import numpy as np

from .exceptions import ImageWriteError

logger = logging.getLogger(__name__)


class ImageWriter:
    def __init__(self):
        logger.debug("初始化 ImageWriter 组件。")

    def write_image(self, image: Image.Image, path: str) -> None:
        """
        保存处理后的图像到指定路径。

        :param image: Pillow Image对象
        :param path: 保存路径的字符串
        """
        logger.info(f"开始保存图像到 {path}。")
        try:
            np_image = np.array(image)
            logger.debug(f"保存图像的数据类型: {np_image.dtype}, 形状: {np_image.shape}")
            image.save(path)
            logger.info(f"成功保存图像到 {path}。")
        except Exception as e:
            logger.error(f"保存图像失败: {path}，错误: {e}")
            raise ImageWriteError(f"Failed to write image to {path}: {e}")
