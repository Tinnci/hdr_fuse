# src/hsv_processing.py

"""
HSV空间处理组件，提供图像在HSV色彩空间中的转换和增强功能。
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List
from .exceptions import ExposureFusionError, ToneMappingError

logger = logging.getLogger(__name__)

class HSVProcessor:
    def __init__(self):
        logger.debug("初始化 HSVProcessor 组件。")

    def convert_to_hsv(self, image: Image.Image) -> np.ndarray:
        """
        将RGB图像转换为HSV色彩空间。

        :param image: Pillow Image对象
        :return: HSV色彩空间的NumPy数组
        """
        logger.debug("将RGB图像转换为HSV色彩空间。")
        rgb = np.array(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return hsv

    def convert_to_rgb(self, hsv_image: np.ndarray) -> Image.Image:
        """
        将HSV图像转换回RGB色彩空间。

        :param hsv_image: HSV色彩空间的NumPy数组
        :return: Pillow Image对象
        """
        logger.debug("将HSV图像转换回RGB色彩空间。")
        rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def enhance_v_channel(self, hsv_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        对HSV图像的V通道进行增强，例如对比度拉伸。

        :param hsv_images: HSV色彩空间的图像列表
        :return: 处理后的HSV图像列表
        """
        logger.info("开始增强V通道。")
        enhanced_images = []
        for idx, hsv in enumerate(hsv_images, start=1):
            logger.debug(f"增强第 {idx} 张HSV图像的V通道。")
            try:
                h, s, v = cv2.split(hsv)
                v = cv2.equalizeHist(v)  # 直方图均衡化增强V通道
                enhanced_hsv = cv2.merge([h, s, v])
                enhanced_images.append(enhanced_hsv)
                logger.debug(f"成功增强第 {idx} 张HSV图像的V通道。")
            except Exception as e:
                logger.error(f"增强第 {idx} 张HSV图像的V通道失败: {e}")
                raise ExposureFusionError(f"Failed to enhance V channel for image {idx}: {e}")
        logger.info("V通道增强完成。")
        return enhanced_images

    def adjust_saturation(self, hsv_image: np.ndarray, saturation_scale: float) -> np.ndarray:
        """
        调整HSV图像的饱和度。

        :param hsv_image: HSV色彩空间的NumPy数组
        :param saturation_scale: 饱和度缩放比例
        :return: 调整后的HSV色彩空间的NumPy数组
        """
        logger.debug(f"调整饱和度，缩放比例: {saturation_scale}")
        h, s, v = cv2.split(hsv_image)
        s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
        adjusted_hsv = cv2.merge([h, s, v])
        logger.debug("饱和度调整完成。")
        return adjusted_hsv

    def adjust_hue(self, hsv_image: np.ndarray, hue_shift: float) -> np.ndarray:
        """
        调整HSV图像的色调。

        :param hsv_image: HSV色彩空间的NumPy数组
        :param hue_shift: 色调偏移量（度）
        :return: 调整后的HSV色彩空间的NumPy数组
        """
        logger.debug(f"调整色调，偏移量: {hue_shift} 度")
        h, s, v = cv2.split(hsv_image)
        # OpenCV中的H通道范围是0-179
        h = np.mod(h.astype(np.float32) + hue_shift / 2, 180).astype(np.uint8)  # 将度转换为OpenCV的范围
        adjusted_hsv = cv2.merge([h, s, v])
        logger.debug("色调调整完成。")
        return adjusted_hsv
