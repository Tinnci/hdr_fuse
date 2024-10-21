# src/image_aligner.py

"""
图像对齐组件，使用OpenCV进行特征点检测、匹配和图像对齐。
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List
from .exceptions import ImageAlignError

logger = logging.getLogger(__name__)

class ImageAligner:
    def __init__(self, feature_detector: str = 'SIFT'):
        """
        初始化图像对齐组件，支持SIFT或ORB特征检测算法。

        :param feature_detector: 'SIFT' 或 'ORB'，用于选择特征点检测算法
        """
        logger.debug(f"初始化 ImageAligner 组件，使用特征检测器: {feature_detector}")
        if feature_detector.upper() == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher_type = cv2.NORM_L2
        elif feature_detector.upper() == 'ORB':
            self.detector = cv2.ORB_create()
            self.matcher_type = cv2.NORM_HAMMING
        else:
            logger.error("不支持的特征检测算法。请选择 'SIFT' 或 'ORB'。")
            raise ValueError("Unsupported feature detector. Use 'SIFT' or 'ORB'.")

    def align_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        对齐图像，返回对齐后的图像列表。

        :param images: 输入的Pillow Image对象列表
        :return: 对齐后的Pillow Image对象列表
        """
        logger.info("开始对齐图像。")
        if not images:
            logger.error("没有提供任何图像进行对齐。")
            raise ImageAlignError("No images provided for alignment.")

        aligned_images = [images[0]]  # 基准图像不需要对齐
        base_image_np = np.array(images[0])

        for idx, img in enumerate(images[1:], start=2):
            logger.debug(f"对齐第 {idx} 张图像。")
            try:
                np_img = np.array(img)
                aligned_np = self._align_two_images(base_image_np, np_img)
                aligned_img = Image.fromarray(aligned_np)
                aligned_images.append(aligned_img)
                logger.debug(f"成功对齐第 {idx} 张图像。")
            except Exception as e:
                logger.error(f"对齐第 {idx} 张图像失败: {e}")
                raise ImageAlignError(f"Failed to align image {idx}: {e}")

        logger.info("所有图像对齐完成。")
        return aligned_images

    def _align_two_images(self, base_image: np.ndarray, img_to_align: np.ndarray) -> np.ndarray:
        """
        使用OpenCV进行特征点匹配和变换矩阵计算来对齐两张图像。

        :param base_image: 基准图像的NumPy数组
        :param img_to_align: 需要对齐的图像的NumPy数组
        :return: 对齐后的图像的NumPy数组
        """
        logger.debug("转换图像为灰度图。")
        gray_base = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
        gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)

        logger.debug("检测和计算特征点。")
        keypoints1, descriptors1 = self.detector.detectAndCompute(gray_base, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(gray_align, None)

        if descriptors1 is None or descriptors2 is None:
            logger.error("无法在其中一张图像中找到描述符。")
            raise ImageAlignError("No descriptors found in one of the images.")

        logger.debug("匹配特征点。")
        matcher = cv2.BFMatcher(self.matcher_type, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        logger.debug(f"找到 {len(matches)} 个匹配点。")
        if len(matches) < 4:
            logger.error("匹配点数量不足，无法计算变换矩阵。")
            raise ImageAlignError("Not enough matches found between images.")

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        logger.debug("计算单应性矩阵。")
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            logger.error("无法计算单应性矩阵。")
            raise ImageAlignError("Homography computation failed.")

        height, width = base_image.shape[:2]
        logger.debug("应用变换矩阵进行图像对齐。")
        aligned_image = cv2.warpPerspective(img_to_align, M, (width, height))
        return aligned_image
