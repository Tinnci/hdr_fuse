# src/hdr_fuse/image_aligner.py

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List
import gc

from .exceptions import ImageAlignError

logger = logging.getLogger(__name__)

class ImageAligner:
    def __init__(self, feature_detector: str = 'SIFT', downscale_factor: float = 1.0):
        """
        初始化图像对齐组件，支持SIFT或ORB特征检测算法，并可选地下采样图像。

        :param feature_detector: 'SIFT' 或 'ORB'，用于选择特征点检测算法
        :param downscale_factor: 下采样比例（0 < downscale_factor <= 1）
        """
        logger.debug(f"初始化 ImageAligner 组件，使用特征检测器: {feature_detector}, 下采样比例: {downscale_factor}")
        if feature_detector.upper() == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher_type = cv2.NORM_L2
            logger.debug("使用 SIFT 特征检测器和 NORM_L2 匹配器。")
        elif feature_detector.upper() == 'ORB':
            self.detector = cv2.ORB_create()
            self.matcher_type = cv2.NORM_HAMMING
            logger.debug("使用 ORB 特征检测器和 NORM_HAMMING 匹配器。")
        else:
            logger.error("不支持的特征检测算法。请选择 'SIFT' 或 'ORB'。")
            raise ValueError("Unsupported feature detector. Use 'SIFT' or 'ORB'.")

        self.downscale_factor = downscale_factor

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

        # 确保所有图像都是 RGB 模式
        logger.debug("确保所有图像都是 RGB 模式。")
        images = [img.convert('RGB') for img in images]

        # 下采样图像
        if self.downscale_factor < 1.0:
            logger.debug(f"下采样图像，比例为 {self.downscale_factor}")
            images = [img.resize(
                (int(img.width * self.downscale_factor), int(img.height * self.downscale_factor)),
                Image.ANTIALIAS
            ) for img in images]

        aligned_images = [images[0]]  # 基准图像不需要对齐
        base_image_np = np.array(images[0])
        logger.debug(f"基准图像数据类型: {base_image_np.dtype}, 形状: {base_image_np.shape}")

        for idx, img in enumerate(images[1:], start=2):
            logger.debug(f"对齐第 {idx} 张图像。")
            try:
                np_img = np.array(img)
                logger.debug(f"待对齐图像数据类型: {np_img.dtype}, 形状: {np_img.shape}")
                aligned_np = self._align_two_images(base_image_np, np_img)
                aligned_img = Image.fromarray(aligned_np)
                aligned_images.append(aligned_img)
                logger.debug(f"成功对齐第 {idx} 张图像。")

                # 释放临时变量
                del np_img, aligned_np
                gc.collect()
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
        try:
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)
            gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            logger.error(f"转换为灰度图失败: {e}")
            raise ImageAlignError(f"Failed to convert images to grayscale: {e}")

        logger.debug(f"基准灰度图数据类型: {gray_base.dtype}, 形状: {gray_base.shape}")
        logger.debug(f"待对齐灰度图数据类型: {gray_align.dtype}, 形状: {gray_align.shape}")

        logger.debug("检测和计算特征点。")
        keypoints1, descriptors1 = self.detector.detectAndCompute(gray_base, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(gray_align, None)
        logger.debug(f"基准图像特征点数量: {len(keypoints1)}, 描述符类型: {descriptors1.dtype if descriptors1 is not None else 'None'}")
        logger.debug(f"待对齐图像特征点数量: {len(keypoints2)}, 描述符类型: {descriptors2.dtype if descriptors2 is not None else 'None'}")

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
        logger.debug(f"源点数量: {src_pts.shape[0]}, 目标点数量: {dst_pts.shape[0]}")

        logger.debug("计算单应性矩阵。")
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            logger.error("无法计算单应性矩阵。")
            raise ImageAlignError("Homography computation failed.")

        height, width = base_image.shape[:2]
        logger.debug(f"基准图像尺寸: width={width}, height={height}")
        logger.debug("应用变换矩阵进行图像对齐。")
        aligned_image = cv2.warpPerspective(img_to_align, M, (width, height))
        logger.debug(f"对齐后的图像数据类型: {aligned_image.dtype}, 形状: {aligned_image.shape}")

        return aligned_image
