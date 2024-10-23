# src/exposure_fusion.py

"""
曝光融合组件，使用HSV色彩空间的V通道进行曝光融合，采用加权平均方法。
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

    def compute_weight(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        计算每张图像的权重图，基于V通道的曝光、对比度和饱和度。

        :param hsv_image: HSV色彩空间的图像
        :return: 权重图
        """
        logger.debug("计算权重图。")
        # 提取各通道
        H, S, V = cv2.split(hsv_image.astype(np.float32))
        
        # 曝光权重：避免过曝和欠曝
        exposure_weight = np.where((V > 0) & (V < 255), 1.0, 0.0)
        
        # 对比度权重：使用V通道的局部对比度（可通过梯度或拉普拉斯算子计算）
        laplacian = cv2.Laplacian(V, cv2.CV_32F)
        contrast_weight = np.abs(laplacian)
        contrast_weight = cv2.normalize(contrast_weight, None, 0, 1, cv2.NORM_MINMAX)
        
        # 饱和度权重：使用S通道
        saturation_weight = S / 255.0
        
        # 综合权重
        weight = exposure_weight * contrast_weight * saturation_weight
        
        # 归一化权重以避免全零
        weight = np.clip(weight, 0, 1)
        
        return weight

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

            # 计算每张图像的权重图
            logger.debug("计算每张图像的权重图。")
            weight_maps = [self.compute_weight(hsv) for hsv in hsv_images]

            # 提取V通道
            logger.debug("提取V通道。")
            v_channels = [hsv[:, :, 2].astype(np.float32) for hsv in hsv_images]

            # 计算加权V通道的总和和权重总和
            logger.debug("计算加权V通道和权重总和。")
            weighted_v = np.zeros_like(v_channels[0], dtype=np.float32)
            weight_sum = np.zeros_like(v_channels[0], dtype=np.float32)

            for v, w in zip(v_channels, weight_maps):
                weighted_v += w * v
                weight_sum += w

            # 避免除以零
            weight_sum = np.where(weight_sum == 0, 1e-5, weight_sum)

            # 归一化融合后的V通道
            logger.debug("归一化融合后的V通道。")
            fused_v = (weighted_v / weight_sum).astype(np.uint8)

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
