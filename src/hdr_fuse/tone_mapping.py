# src/tone_mapping.py

"""
色调映射组件，使用OpenCV的色调映射算法将HDR图像转换为LDR图像。
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional
from .exceptions import ToneMappingError

logger = logging.getLogger(__name__)

class ToneMapper:
    def __init__(self, method: str = 'Reinhard', gamma: float = 1.0):
        """
        初始化色调映射组件，支持不同的色调映射方法。

        :param method: 色调映射方法，例如 'Reinhard'、'Drago' 等
        :param gamma: gamma校正值
        """
        logger.debug(f"初始化 ToneMapper 组件，方法: {method}, gamma: {gamma}")
        self.method = method.lower()
        self.gamma = gamma
        if self.method == 'reinhard':
            self.mapper = cv2.createTonemapReinhard(gamma=gamma)
        elif self.method == 'drago':
            self.mapper = cv2.createTonemapDrago(gamma=gamma)
        elif self.method == 'durand':
            self.mapper = cv2.createTonemapDurand(gamma=gamma)
        else:
            logger.error("不支持的色调映射方法。请选择 'Reinhard'、'Drago' 或 'Durand'。")
            raise ValueError("Unsupported tone mapping method. Use 'Reinhard', 'Drago', or 'Durand'.")

    def map_tone(self, image: Image.Image) -> Image.Image:
        """
        对融合后的图像进行色调映射，返回色调映射后的图像。

        :param image: 融合后的Pillow Image对象
        :return: 色调映射后的Pillow Image对象
        """
        logger.info("开始进行色调映射。")
        try:
            np_image = np.array(image).astype(np.float32) / 255.0
            logger.debug("应用色调映射算法。")
            ldr = self.mapper.process(np_image)
            ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
            tone_mapped_image = Image.fromarray(ldr)
            logger.info("色调映射完成。")
            return tone_mapped_image
        except Exception as e:
            logger.error(f"色调映射过程中出现错误: {e}")
            raise ToneMappingError(f"Failed during tone mapping: {e}")
