# src/hdr_fuse/exposure_detection.py

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def estimate_exposure_level(image: Image.Image) -> float:
    """
    估计图像的曝光等级，通过计算图像的平均亮度。
    
    :param image: Pillow Image对象
    :return: 曝光等级（较高值表示曝光较高）
    """
    logger.debug("估计图像的曝光等级。")
    np_image = np.array(image)
    logger.debug(f"输入图像数据类型: {np_image.dtype}, 形状: {np_image.shape}")
    if np_image.dtype != np.float32 and np_image.dtype != np.float64:
        logger.debug("转换图像数据类型为float32并归一化到[0.0, 1.0]")
        np_image = np_image.astype(np.float32) / 255.0
    else:
        logger.debug("图像已为浮点类型，假设归一化到[0.0, 1.0]")
    logger.debug(f"图像归一化后数据类型: {np_image.dtype}, 范围: {np_image.min()}-{np_image.max()}")
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    logger.debug(f"灰度图像数据类型: {gray.dtype}, 范围: {gray.min()}-{gray.max()}")
    avg_luminance = np.mean(gray)
    logger.debug(f"图像平均亮度: {avg_luminance}")
    return avg_luminance
