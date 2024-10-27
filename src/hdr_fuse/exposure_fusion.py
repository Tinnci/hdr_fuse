# src/exposure_fusion.py

"""
曝光融合组件，使用HSV色彩空间的V通道进行曝光融合。
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List
from .exceptions import ExposureFusionError

logger = logging.getLogger(__name__)

class ExposureFusion:
    def __init__(self):
        logger.debug("初始化 ExposureFusion 组件。")

    def fuse_images(self, images: List[Image.Image]) -> Image.Image:
        """
        融合多张对齐后的图像，返回融合后的图像。

        :param images: 对齐后的Pillow Image对象列表
        :return: 融合后的Pillow Image对象
        """
        logger.info("开始进行曝光融合。")
        if not images:
            logger.error("没有提供任何图像进行曝光融合。")
            raise ExposureFusionError("No images provided for fusion.")

        try:
            # 将Pillow图像转换为NumPy数组并转换到HSV空间
            logger.debug("将图像转换到HSV色彩空间。")
            np_images = [np.array(img) for img in images]
            hsv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in np_images]

            # 提取V通道并进行融合
            logger.debug("提取V通道并进行融合。")
            v_channels = [hsv[:, :, 2].astype(np.float32) for hsv in hsv_images]
            fused_v = np.mean(v_channels, axis=0).astype(np.uint8)

            # 使用融合后的V通道与第一个图像的H和S通道组合
            logger.debug("合并融合后的V通道与基准图像的H和S通道。")
            base_hsv = hsv_images[0].copy()
            base_hsv[:, :, 2] = fused_v
            fused_hsv = base_hsv

            # 转换回RGB
            logger.debug("将融合后的HSV图像转换回RGB色彩空间。")
            fused_rgb = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2RGB)
            fused_image = Image.fromarray(fused_rgb)
            logger.info("曝光融合完成。")
            return fused_image
        except Exception as e:
            logger.error(f"曝光融合过程中出现错误: {e}")
            raise ExposureFusionError(f"Failed during exposure fusion: {e}")
