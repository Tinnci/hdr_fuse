import logging
from PIL import Image
from typing import List
import cv2
import numpy as np

from .exceptions import ImageReadError

logger = logging.getLogger(__name__)

class ImageReader:
    def __init__(self):
        logger.debug("初始化 ImageReader 组件。")

    def read_images(self, paths: List[str], apply_noise_reduction: bool = False) -> List[Image.Image]:
        """
        读取多个图像文件，返回Pillow的Image对象列表。
        
        :param paths: 图像文件路径的字符串列表
        :param apply_noise_reduction: 是否应用降噪处理，默认为 False
        :return: Pillow的Image对象列表
        """
        logger.info(f"开始读取 {len(paths)} 张图像。")
        images = []
        for path in paths:
            try:
                logger.debug(f"读取图像: {path}")
                img = Image.open(path).convert('RGB')  # 确保图像为RGB格式
                if apply_noise_reduction:
                    img = self.reduce_noise(img)
                logger.debug(f"成功读取图像: {path}，大小: {img.size}, 模式: {img.mode}")
                images.append(img)
            except Exception as e:
                logger.error(f"读取图像失败: {path}，错误: {e}")
                raise ImageReadError(f"Failed to read image at {path}: {e}")
        logger.info(f"成功读取了 {len(images)} 张图像。")
        return images

    def reduce_noise(self, image: Image.Image) -> Image.Image:
        """
        Applies noise reduction to the input image using OpenCV's fastNlMeansDenoisingColored.
        
        :param image: Pillow Image object to process
        :return: Pillow Image object with reduced noise
        """
        logger.info("Applying noise reduction to image.")
        np_image = np.array(image)
        denoised_image = cv2.fastNlMeansDenoisingColored(np_image, None, 10, 10, 7, 21)
        return Image.fromarray(denoised_image)
