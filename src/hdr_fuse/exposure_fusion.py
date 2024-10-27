# src/hdr_fuse/exposure_fusion.py

"""
曝光融合组件，支持多种融合方法，包括平均融合、Mertens算法、金字塔融合以及去鬼影。
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional

from .exceptions import ExposureFusionError

logger = logging.getLogger(__name__)

class ExposureFusion:
    def __init__(self, method: str = 'average'):
        """
        初始化曝光融合组件，支持不同的融合方法。

        :param method: 'average'、'mertens'、'pyramid' 或 'ghost_removal'，用于选择曝光融合算法
        """
        logger.debug(f"初始化 ExposureFusion 组件，方法: {method}")
        self.method = method.lower()
        if self.method == 'mertens':
            # 使用Mertens融合器，设置对比度权重、饱和度权重和曝光权重
            self.fuser = cv2.createMergeMertens(
                contrast_weight=1.0,
                saturation_weight=1.0,
                exposure_weight=1.0
            )
            logger.debug("使用 Mertens 融合器，已设置默认权重。")
        elif self.method in ['average', 'pyramid', 'ghost_removal']:
            self.fuser = None  # 使用自定义融合
            logger.debug("使用自定义融合方法.")
        else:
            logger.error("不支持的曝光融合方法。请选择 'average'、'mertens'、'pyramid' 或 'ghost_removal'。")
            raise ValueError("Unsupported fusion method. Use 'average', 'mertens', 'pyramid', or 'ghost_removal'.")

    def fuse_images(self, images: List[Image.Image], exposure_levels: Optional[List[float]] = None) -> Image.Image:
        """
        融合多张对齐后的图像，返回融合后的图像。

        :param images: 对齐后的Pillow Image对象列表
        :param exposure_levels: 每张图像的曝光等级，用于加权融合（可选）
        :return: 融合后的Pillow Image对象
        """
        logger.info("开始进行曝光融合。")
        if not images:
            logger.error("没有提供任何图像进行曝光融合。")
            raise ExposureFusionError("No images provided for fusion.")

        try:
            logger.debug(f"共有 {len(images)} 张图像需要融合。")
            # 将图像转换为 NumPy 数组并归一化
            np_images = [np.array(img).astype(np.float32) / 255.0 for img in images]
            logger.debug("图像已转换为float32并归一化到[0.0, 1.0]。")

            if self.method == 'mertens':
                logger.debug("使用Mertens算法进行曝光融合。")
                # 将RGB图像转换为BGR格式
                bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in np_images]
                logger.debug("图像已转换为BGR色彩空间。")
                
                # 处理输入图像以确保它们在有效范围内
                bgr_images = [np.clip(img, 0.0, 1.0) for img in bgr_images]
                logger.debug("确保输入图像在[0.0, 1.0]范围内")
                
                # 执行Mertens融合
                fused = self.fuser.process(bgr_images)
                logger.debug(f"Mertens融合后的图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")
                
                # 确保融合结果在有效范围内
                fused = np.clip(fused, 0.0, 1.0)
                logger.debug(f"裁剪后的融合图像范围: {fused.min()}-{fused.max()}")
                
                # 将融合后的图像转换回RGB格式
                fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)
                logger.debug("融合后的图像已转换回RGB色彩空间。")
                
                # 线性拉伸以增加对比度
                if fused.max() - fused.min() > 0:
                    fused = (fused - fused.min()) / (fused.max() - fused.min())
                    logger.debug("应用线性拉伸增加对比度。")
                
                # 转换为8位图像
                fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
                logger.debug("Mertens融合完成，图像已转换回uint8。")
                logger.debug(f"Mertens融合后的图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")
            
            elif self.method == 'average':
                logger.debug("使用自定义平均算法进行曝光融合。")
                fused = np.mean(np.stack(np_images), axis=0)
                fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
                logger.debug("平均融合完成，图像已转换回uint8。")
                logger.debug(f"平均融合后的图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")
            elif self.method == 'pyramid':
                logger.debug("使用金字塔融合算法进行曝光融合。")
                fused = self.pyramid_fusion(np_images)
                fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
                logger.debug("金字塔融合完成，图像已转换回uint8。")
                logger.debug(f"金字塔融合后的图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")
            elif self.method == 'ghost_removal':
                logger.debug("使用去鬼影算法进行曝光融合。")
                if exposure_levels is None:
                    logger.error("去鬼影融合需要 exposure_levels 参数。")
                    raise ExposureFusionError("Exposure levels required for ghost removal fusion.")
                fused = self.ghost_removal_fusion(np_images, exposure_levels)
                fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
                logger.debug("去鬼影融合完成，图像已转换回uint8。")
                logger.debug(f"去鬼影融合后的图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")
            else:
                logger.error(f"未知的融合方法: {self.method}")
                raise ExposureFusionError(f"Unknown fusion method: {self.method}")

            fused_image = Image.fromarray(fused, 'RGB')
            logger.debug(f"融合后的Pillow Image数据类型: {fused_image.mode}, 尺寸: {fused_image.size}")
            logger.info("曝光融合完成。")
            return fused_image

        except Exception as e:
            logger.error(f"曝光融合过程中出现错误: {e}")
            raise ExposureFusionError(f"Failed during exposure fusion: {e}")

    def pyramid_fusion(self, np_images: List[np.ndarray]) -> np.ndarray:
        """
        使用金字塔融合算法融合多张图像。

        :param np_images: 图像的NumPy数组列表
        :return: 融合后的图像数组
        """
        logger.debug("开始金字塔融合。")

        try:
            # 确保所有图像尺寸一致
            min_height = min(img.shape[0] for img in np_images)
            min_width = min(img.shape[1] for img in np_images)
            
            # 确保尺寸是2的倍数
            min_width = min_width - (min_width % 2)
            min_height = min_height - (min_height % 2)
            
            logger.debug(f"调整后的目标尺寸: 宽度={min_width}, 高度={min_height}")
            
            # 调整所有图像到相同尺寸
            np_images_resized = [cv2.resize(img, (min_width, min_height), 
                                        interpolation=cv2.INTER_LINEAR) 
                            for img in np_images]
            
            # 计算合适的金字塔层数
            num_levels = min(
                int(np.log2(min_width)) - 4,  # 确保最小宽度不小于16
                int(np.log2(min_height)) - 4,  # 确保最小高度不小于16
                4  # 最大层数限制
            )
            num_levels = max(1, num_levels)  # 至少保证1层
            logger.debug(f"计算得到的金字塔层数: {num_levels}")

            # 为每个图像构建高斯金字塔和拉普拉斯金字塔
            gaussian_pyramids = []
            laplacian_pyramids = []
            
            for idx, img in enumerate(np_images_resized):
                # 构建高斯金字塔
                gaussian = [img.copy()]
                for level in range(num_levels):
                    # 确保当前图像尺寸是2的倍数
                    current = gaussian[-1]
                    if current.shape[0] % 2 == 1:
                        current = current[:-1, :, :]
                    if current.shape[1] % 2 == 1:
                        current = current[:, :-1, :]
                    next_level = cv2.pyrDown(current)
                    gaussian.append(next_level)
                gaussian_pyramids.append(gaussian)

                # 构建拉普拉斯金字塔
                laplacian = []
                for level in range(num_levels):
                    size = (gaussian[level].shape[1], gaussian[level].shape[0])
                    expanded = cv2.pyrUp(gaussian[level + 1], dstsize=size)
                    # 确保尺寸匹配
                    if expanded.shape != gaussian[level].shape:
                        expanded = cv2.resize(expanded, (gaussian[level].shape[1], 
                                                    gaussian[level].shape[0]))
                    diff = cv2.subtract(gaussian[level], expanded)
                    laplacian.append(diff)
                # 添加最顶层
                laplacian.append(gaussian[-1])
                laplacian_pyramids.append(laplacian)
                
                logger.debug(f"图像 {idx+1} 的金字塔构建完成")

            # 融合每一层
            fused_pyramid = []
            for level in range(num_levels + 1):  # +1 是因为包含了顶层
                # 获取当前层的所有图像
                current_level = [pyramid[level] for pyramid in laplacian_pyramids]
                # 融合当前层
                fused_level = np.mean(current_level, axis=0)
                fused_pyramid.append(fused_level)
                logger.debug(f"第 {level+1} 层融合完成，尺寸: {fused_level.shape}")

            # 重建融合图像
            fused = fused_pyramid[-1]  # 从顶层开始
            for level in range(num_levels - 1, -1, -1):
                size = (fused_pyramid[level].shape[1], fused_pyramid[level].shape[0])
                # 使用指定尺寸进行上采样
                fused = cv2.pyrUp(fused, dstsize=size)
                # 确保尺寸匹配
                if fused.shape != fused_pyramid[level].shape:
                    fused = cv2.resize(fused, (fused_pyramid[level].shape[1], 
                                            fused_pyramid[level].shape[0]))
                fused = cv2.add(fused, fused_pyramid[level])
                logger.debug(f"重建层 {level+1}，当前尺寸: {fused.shape}")

            # 确保值域正确
            fused = np.clip(fused, 0, 1)
            
            logger.debug("金字塔融合完成。")
            logger.debug(f"最终融合图像尺寸: {fused.shape}, 范围: [{fused.min()}, {fused.max()}]")
            
            return fused

        except Exception as e:
            logger.error(f"金字塔融合过程中出现错误: {e}")
            raise ExposureFusionError(f"Pyramid fusion failed: {e}")

    def calculate_pyramid_levels(self, width: int, height: int) -> int:
        """
        动态计算金字塔层数，确保图像尺寸足够且可被2整除。

        :param width: 图像宽度
        :param height: 图像高度
        :return: 金字塔层数
        """
        levels = 0
        while width >= 32 and height >= 32 and levels < 6:
            if width % 2 != 0 or height % 2 != 0:
                break
            width = width // 2
            height = height // 2
            levels += 1
        logger.debug(f"动态计算的金字塔层数: {levels}")
        return levels

    def ghost_removal_fusion(self, np_images: List[np.ndarray], exposure_levels: List[float]) -> np.ndarray:
        """
        使用高级去鬼影算法融合多张图像。

        :param np_images: 图像的NumPy数组列表
        :param exposure_levels: 每张图像的曝光等级
        :return: 融合后的图像数组
        """
        logger.debug("开始高级去鬼影融合。")
        try:
            # 计算权重（曝光等级归一化）
            weights = np.array(exposure_levels)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                logger.error("曝光等级之和为零，无法归一化权重。")
                raise ExposureFusionError("Sum of exposure levels is zero.")
            weights = weights / weights_sum
            weights = weights[:, np.newaxis, np.newaxis, np.newaxis]  # 适应图像维度
            logger.debug(f"归一化后的权重: {weights.flatten()}")

            # 初始化融合图像和掩膜
            fused = np.zeros_like(np_images[0], dtype=np.float32)
            mask = np.zeros_like(np_images[0], dtype=np.float32)

            for idx, (img, weight) in enumerate(zip(np_images, weights)):
                logger.debug(f"处理图像 {idx+1}，权重: {weight}")
                # 使用权重应用图像
                fused += img * weight

                # 计算差异掩膜以检测运动物体
                if idx > 0:
                    diff = cv2.absdiff(np_images[0], img)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    _, motion_mask = cv2.threshold(gray_diff, 0.05, 1.0, cv2.THRESH_BINARY)
                    mask += motion_mask[:, :, np.newaxis] * weight
                    logger.debug(f"图像 {idx+1} 的运动掩膜平均值: {np.mean(motion_mask)}")

            # 归一化掩膜
            mask = np.clip(mask, 0, 1)
            logger.debug(f"掩膜数据类型: {mask.dtype}, 范围: {mask.min()}-{mask.max()}")

            # 应用掩膜到融合图像
            fused = np.clip(fused * (1 - mask) + np_images[0].astype(np.float32) * mask, 0, 1).astype(np.float32)
            logger.debug(f"应用掩膜后的融合图像数据类型: {fused.dtype}, 范围: {fused.min()}-{fused.max()}")

            # 处理可能的NaN或Inf值
            fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
            logger.debug("处理NaN和Inf值。")

            # 转回RGB格式
            fused_rgb = fused  # Assuming images are in RGB
            logger.debug("高级去鬼影融合完成。")
            return fused_rgb

        except Exception as e:
            logger.error(f"高级去鬼影融合过程中出现错误: {e}")
            raise ExposureFusionError(f"Advanced ghost removal fusion failed: {e}")
