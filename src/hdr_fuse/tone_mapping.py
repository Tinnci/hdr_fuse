# src/hdr_fuse/tone_mapping.py

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional

from .exceptions import ToneMappingError

logger = logging.getLogger(__name__)


class ToneMapper:
    def __init__(self, method: str = 'reinhard', gamma: Optional[float] = None):
        """
        初始化色调映射组件，支持不同的色调映射方法。

        :param method: 色调映射方法，例如 'Reinhard'、'Drago' 等
        :param gamma: gamma校正值，如果为None，则根据图像动态计算
        """
        logger.debug(f"初始化 ToneMapper 组件，方法: {method}, gamma: {gamma}")
        self.method = method.lower()
        self.gamma = gamma if gamma is not None else 1.0  # 默认Gamma为1.0
        if self.method == 'reinhard':
            self.mapper = cv2.createTonemapReinhard(gamma=self.gamma)
            logger.debug("使用 Reinhard 色调映射器。")
        elif self.method == 'drago':
            self.mapper = cv2.createTonemapDrago(gamma=self.gamma)
            logger.debug("使用 Drago 色调映射器。")
        elif self.method == 'durand':
            self.mapper = cv2.createTonemapDurand(gamma=self.gamma)
            logger.debug("使用 Durand 色调映射器。")
        else:
            logger.error("不支持的色调映射方法。请选择 'Reinhard'、'Drago' 或 'Durand'。")
            raise ValueError("Unsupported tone mapping method. Use 'Reinhard', 'Drago', or 'Durand'.")

    def map_tone(self, image: Image.Image, dynamic_gamma: bool = False) -> Image.Image:
        """
        对融合后的图像进行色调映射，返回色调映射后的图像。

        :param image: 融合后的Pillow Image对象
        :param dynamic_gamma: 是否根据图像动态调整Gamma值
        :return: 色调映射后的Pillow Image对象
        """
        logger.info("开始进行色调映射。")
        try:
            np_image = np.array(image).astype(np.float32) / 255.0
            logger.debug(f"输入图像数据类型: {np_image.dtype}, 范围: {np_image.min()}-{np_image.max()}, 形状: {np_image.shape}")

            if dynamic_gamma:
                # 根据图像的平均亮度调整Gamma值
                avg_luminance = np.mean(np_image)
                # 设定Gamma值的范围，例如从0.5到2.2
                gamma = np.interp(avg_luminance, [0, 1], [2.2, 0.5])
                # 确保Gamma值在合理范围内
                gamma = max(0.5, min(gamma, 2.2))
                logger.debug(f"动态调整Gamma值为: {gamma}")
                self.mapper = self._update_gamma(gamma)

            logger.debug("应用色调映射算法。")
            ldr = self.mapper.process(np_image)
            logger.debug(f"色调映射后图像数据类型: {ldr.dtype}, 范围: {ldr.min()}-{ldr.max()}")

            # 处理可能的NaN或Inf值
            if np.isnan(ldr).any() or np.isinf(ldr).any():
                logger.warning("色调映射后图像存在NaN或Inf值，进行处理。")
                ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
                logger.debug("处理NaN和Inf值。")

            ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
            logger.debug(f"色调映射后的图像转换为uint8，范围: {ldr.min()}-{ldr.max()}")

            tone_mapped_image = Image.fromarray(ldr)
            logger.debug(f"色调映射后的图像数据类型: {np.array(tone_mapped_image).dtype}, 形状: {np.array(tone_mapped_image).shape}")
            logger.info("色调映射完成。")
            return tone_mapped_image
        except Exception as e:
            logger.error(f"色调映射过程中出现错误: {e}")
            raise ToneMappingError(f"Failed during tone mapping: {e}")

    def _update_gamma(self, gamma: float):
        if self.method == 'reinhard':
            logger.debug(f"更新 Reinhard 色调映射器的 Gamma 值为: {gamma}")
            return cv2.createTonemapReinhard(gamma=gamma)
        elif self.method == 'drago':
            logger.debug(f"更新 Drago 色调映射器的 Gamma 值为: {gamma}")
            return cv2.createTonemapDrago(gamma=gamma)
        elif self.method == 'durand':
            logger.debug(f"更新 Durand 色调映射器的 Gamma 值为: {gamma}")
            return cv2.createTonemapDurand(gamma=gamma)
