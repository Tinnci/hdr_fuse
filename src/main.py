# src/main.py

"""
主函数，协调各个组件完成HDR多帧合成处理流程（支持批处理和单个图像集处理）。
"""

import gc
import os
import argparse
import logging
import sys
from typing import List, Tuple, Optional
import multiprocessing
from multiprocessing import Pool, Manager

import cv2
import numpy as np

from hdr_fuse.exposure_detection import estimate_exposure_level
from hdr_fuse.image_reader import ImageReader
from hdr_fuse.image_aligner import ImageAligner
from hdr_fuse.exposure_fusion import ExposureFusion
from hdr_fuse.tone_mapping import ToneMapper
from hdr_fuse.image_writer import ImageWriter
from hdr_fuse.hsv_processing import HSVProcessor
from hdr_fuse.exceptions import (
    ImageReadError,
    ImageAlignError,
    ExposureFusionError,
    ToneMappingError,
    ImageWriteError,
)


def setup_logging(log_level: str):
    """
    设置全局日志配置。
    根据传入的log_level参数设置日志等级。
    """
    logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"无效的日志等级: {log_level}")
        numeric_level = logging.INFO  # 默认等级
    logger.setLevel(numeric_level)

    # 创建控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)

    # 创建文件处理器
    fh = logging.FileHandler("hdr_pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件处理器始终记录DEBUG及以上等级

    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 移除之前的处理器，防止重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 添加处理器到日志器
    logger.addHandler(ch)
    logger.addHandler(fh)


def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Python HDR 多帧合成管线（支持批处理和单个图像集处理）"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入主文件夹路径，包含多个子文件夹，每个子文件夹包含一组不同曝光等级的照片，或者直接包含一组不同曝光等级的照片。",
    )
    parser.add_argument(
        "-f",
        "--feature_detector",
        choices=["SIFT", "ORB"],
        default="SIFT",
        help="特征点检测算法",
    )
    parser.add_argument(
        "-t",
        "--tone_mapping",
        choices=["Reinhard", "Drago", "Durand"],
        default="Reinhard",
        help="色调映射算法",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="色调映射gamma值",
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        choices=["average", "mertens", "pyramid", "ghost_removal"],
        default="average",
        help="曝光融合方法",
    )
    parser.add_argument(
        "--dynamic_gamma",
        action='store_true',
        help="启用动态Gamma调整",
    )
    parser.add_argument(
        "--saturation_scale",
        type=float,
        default=1.0,
        help="饱和度缩放比例",
    )
    parser.add_argument(
        "--hue_shift",
        type=float,
        default=0.0,
        help="色调偏移量（度）",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="设置日志输出等级",
    )
    parser.add_argument(
        "--downscale_factor",
        type=float,
        default=1.0,
        help="图像下采样比例 (0 < factor <= 1)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="并行处理的进程数量",
    )
    parser.add_argument(
        "--noise_reduction",
        action='store_true',
        help="启用降噪处理",
    )
    return parser.parse_args()


def get_subdirectories(root_dir: str, exclude: List[str] = None) -> List[str]:
    """
    获取主文件夹下的所有子文件夹路径，排除指定的文件夹。

    :param root_dir: 主文件夹路径
    :param exclude: 需要排除的子文件夹名称列表
    :return: 子文件夹路径列表
    """
    if exclude is None:
        exclude = []
    subdirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in exclude
    ]
    return subdirs


def get_image_paths(input_dir: str) -> List[str]:
    """
    获取指定文件夹中的所有原始图像文件路径，排除已处理过的_fused.jpg文件。

    :param input_dir: 子文件夹路径
    :return: 图像文件路径列表
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
        and os.path.isfile(os.path.join(input_dir, f))
        and "_fused" not in f.lower()  # 排除包含 _fused 的文件
    ]
    return image_paths


def generate_output_paths(
    subdir_path: str, subdir_name: str, output_root: str
) -> Tuple[str, str]:
    """
    生成输出文件的路径。

    :param subdir_path: 输入子文件夹路径
    :param subdir_name: 子文件夹名称（用于命名输出文件）
    :param output_root: 集中输出文件夹路径
    :return: 输出文件路径元组 (子文件夹内输出路径, 集中输出文件夹内输出路径)
    """
    # 输出文件保存在input_dir下，命名为 subdir_name_fused.jpg
    output_in_subdir = os.path.join(subdir_path, f"{subdir_name}_fused.jpg")

    # 输出文件保存在output_root下，命名为 subdir_name.jpg
    output_in_output = os.path.join(output_root, f"{subdir_name}.jpg")

    return output_in_subdir, output_in_output


def initialize_components(config_params):
    """
    在子进程中初始化所需的组件。

    :param config_params: 配置参数字典
    :return: 初始化后的组件实例
    """
    reader = ImageReader()
    aligner = ImageAligner(
        feature_detector=config_params['feature_detector'],
        downscale_factor=config_params['downscale_factor']
    )
    hsv_processor = HSVProcessor()
    fusion = ExposureFusion(method=config_params['fusion_method'])
    tone_mapper = ToneMapper(
        method=config_params['tone_mapping'],
        gamma=config_params['gamma']
    )
    writer = ImageWriter()
    return reader, aligner, hsv_processor, fusion, tone_mapper, writer


def process_subdirectory(
    args_tuple: Tuple[str, str, str, dict]
):
    """
    处理单个子文件夹中的图像集。

    :param args_tuple: 包含以下元素的元组
        subdir_path: 子文件夹路径
        subdir_name: 子文件夹名称
        output_root: 集中输出文件夹路径
        config_params: 配置参数字典，包括所有必要的参数
    """
    subdir_path, subdir_name, output_root, config_params = args_tuple
    logger = logging.getLogger("Main")
    logger.info(f"开始处理子文件夹: {subdir_path}")

    # 在子进程内部初始化需要的对象
    try:
        reader, aligner, hsv_processor, fusion, tone_mapper, writer = initialize_components(config_params)
    except Exception as e:
        logger.exception(f"初始化组件时发生错误: {e}")
        return

    saturation_scale = config_params['saturation_scale']
    hue_shift = config_params['hue_shift']
    dynamic_gamma = config_params['dynamic_gamma']
    apply_noise_reduction = config_params.get('noise_reduction', False)  # Check noise reduction setting


    image_paths = get_image_paths(subdir_path)
    logger.debug(f"子文件夹 {subdir_path} 中找到的图像文件: {image_paths}")
    if not image_paths:
        logger.warning(f"子文件夹中未找到任何图像文件: {subdir_path}")
        return

    # 初始化变量
    images = []
    aligned_images = []
    exposure_levels = []
    fused_image = None
    hsv_image = None
    enhanced_rgb = None
    tone_mapped_image = None

    try:
        # 步骤 1: 读取图像
        logger.info(f"步骤 1: 读取图像 ({len(image_paths)} 张)。")
        images = reader.read_images(image_paths, apply_noise_reduction=apply_noise_reduction)
        logger.info(f"成功读取了 {len(images)} 张图像。")

        # 步骤 2: 对齐图像
        logger.info("步骤 2: 对齐图像。")
        aligned_images = aligner.align_images(images)
        logger.info("图像对齐完成。")

        # 步骤 3: 估计曝光等级
        logger.info("步骤 3: 估计曝光等级。")
        exposure_levels = [estimate_exposure_level(img) for img in aligned_images]
        for idx, level in enumerate(exposure_levels, start=1):
            logger.info(f"图像 {idx} 的曝光等级: {level:.4f}")

        # 步骤 4: 曝光融合
        logger.info("步骤 4: 曝光融合。")
        logger.info(f"使用的曝光融合方法: {fusion.method}")
        if fusion.method == 'ghost_removal':
            fused_image = fusion.fuse_images(aligned_images, exposure_levels)
            logger.debug("传递参数: exposure_levels")
        else:
            fused_image = fusion.fuse_images(aligned_images)
        logger.info("曝光融合完成。")

        # 步骤 5: 亮度和对比度增强（HSV处理）
        logger.info("步骤 5: 亮度和对比度增强。")
        logger.debug("将融合后的图像转换到HSV色彩空间。")
        hsv_image = hsv_processor.convert_to_hsv(fused_image)
        logger.debug(f"HSV图像数据类型: {hsv_image.dtype}, 形状: {hsv_image.shape}")
        h, s, v = cv2.split(hsv_image)
        logger.debug(f"分离后的通道数据类型: H-{h.dtype}, S-{s.dtype}, V-{v.dtype}")

        # 添加更多调试信息：V通道统计（未均衡化）
        logger.debug(f"V通道统计 - 预均衡化: min={v.min()}, max={v.max()}, mean={v.mean():.4f}, std={v.std():.4f}")

        logger.debug("应用直方图均衡化到V通道。")
        v = cv2.equalizeHist(v)  # 直方图均衡化增强V通道

        # 添加更多调试信息：V通道统计（均衡化后）
        logger.debug(f"V通道统计 - 均衡化后: min={v.min()}, max={v.max()}, mean={v.mean():.4f}, std={v.std():.4f}")

        logger.debug("合并增强后的HSV通道。")
        enhanced_hsv = cv2.merge([h, s, v])

        # 添加调试信息：合并后的HSV图像统计
        logger.debug(f"合并后的HSV图像统计: H-min={h.min()}, H-max={h.max()}, "
                     f"S-min={s.min()}, S-max={s.max()}, V-min={v.min()}, V-max={v.max()}")

        logger.debug("将增强后的HSV图像转换回RGB色彩空间。")
        enhanced_rgb = hsv_processor.convert_to_rgb(enhanced_hsv)
        logger.debug(f"增强后的RGB图像数据类型: {np.array(enhanced_rgb).dtype}, 形状: {np.array(enhanced_rgb).shape}")

        # 添加更多调试信息：增强后的RGB图像统计
        enhanced_rgb_np = np.array(enhanced_rgb)
        logger.debug(f"增强后的RGB图像统计: min={enhanced_rgb_np.min()}, max={enhanced_rgb_np.max()}, "
                     f"mean={enhanced_rgb_np.mean():.4f}, std={enhanced_rgb_np.std():.4f}")

        logger.info("亮度和对比度增强完成。")

        # 步骤 6: 色调映射
        logger.info("步骤 6: 色调映射。")
        logger.info(f"使用的色调映射方法: {tone_mapper.method}, Gamma值: {tone_mapper.gamma}")
        logger.debug("应用色调映射。")
        tone_mapped_image = tone_mapper.map_tone(enhanced_rgb, dynamic_gamma=dynamic_gamma)
        logger.debug(f"色调映射后的图像数据类型: {np.array(tone_mapped_image).dtype}, 形状: {np.array(tone_mapped_image).shape}")
        logger.info("色调映射完成。")

        # 步骤 7: 色彩调整（可选）
        if saturation_scale != 1.0 or hue_shift != 0.0:
            logger.info("步骤 7: 色彩调整。")
            logger.info(f"饱和度缩放比例: {saturation_scale}, 色调偏移量: {hue_shift} 度")
            logger.debug("将色调映射后的图像转换到HSV色彩空间进行调整。")
            hsv_adjusted = hsv_processor.convert_to_hsv(tone_mapped_image)
            logger.debug(f"HSV调整前图像数据类型: {hsv_adjusted.dtype}, 形状: {hsv_adjusted.shape}")

            if saturation_scale != 1.0:
                logger.debug(f"调整饱和度，缩放比例: {saturation_scale}")
                hsv_adjusted = hsv_processor.adjust_saturation(
                    hsv_adjusted, saturation_scale
                )
                logger.debug(f"饱和度调整后的数据类型: {hsv_adjusted.dtype}, 范围: S-{hsv_adjusted[:, :, 1].min()}-{hsv_adjusted[:, :, 1].max()}")

            if hue_shift != 0.0:
                logger.debug(f"调整色调，偏移量: {hue_shift} 度")
                hsv_adjusted = hsv_processor.adjust_hue(hsv_adjusted, hue_shift)
                logger.debug(f"色调调整后的数据类型: {hsv_adjusted.dtype}, 范围: H-{hsv_adjusted[:, :, 0].min()}-{hsv_adjusted[:, :, 0].max()}")

            logger.debug("将调整后的HSV图像转换回RGB色彩空间。")
            tone_mapped_image = hsv_processor.convert_to_rgb(hsv_adjusted)
            logger.debug(f"色彩调整后的RGB图像数据类型: {np.array(tone_mapped_image).dtype}, 形状: {np.array(tone_mapped_image).shape}")
            logger.info("色彩调整完成。")

        # 生成输出路径
        output_in_subdir, output_in_output = generate_output_paths(
            subdir_path, subdir_name, output_root
        )
        logger.debug(f"输出路径 - 子文件夹内: {output_in_subdir}, 集中输出文件夹内: {output_in_output}")

        # 步骤 8: 保存输出图像到子文件夹
        logger.info(f"步骤 8: 保存输出图像到子文件夹: {output_in_subdir}")
        writer.write_image(tone_mapped_image, output_in_subdir)

        # 步骤 9: 保存输出图像到集中输出文件夹
        logger.info(f"步骤 9: 保存输出图像到集中输出文件夹: {output_in_output}")
        writer.write_image(tone_mapped_image, output_in_output)

        logger.info(f"成功处理子文件夹: {subdir_path}")

    except (
        ImageReadError,
        ImageAlignError,
        ExposureFusionError,
        ToneMappingError,
        ImageWriteError,
    ) as e:
        logger.error(f"处理子文件夹 {subdir_path} 过程中出现错误: {e}")
    except Exception as e:
        logger.exception(f"处理子文件夹 {subdir_path} 过程中发生未处理的错误: {e}")
    finally:
        # 释放内存
        for var in [images, aligned_images, exposure_levels, fused_image, hsv_image, enhanced_rgb, tone_mapped_image]:
            if var is not None:
                del var
        gc.collect()


def main():
    """
    主函数，执行HDR多帧合成处理流程（支持批处理和单个图像集处理）。
    """
    args = parse_arguments()
    setup_logging(args.log_level)  # 设置日志等级

    logger = logging.getLogger("Main")
    logger.info("HDR多帧合成处理系统启动。")

    input_root = args.input

    if not os.path.isdir(input_root):
        logger.error(f"输入路径不是一个有效的文件夹: {input_root}")
        sys.exit(1)

    # 检查输入目录是否直接包含图像文件
    logger.debug("检查输入目录是否包含图像文件。")
    image_paths = get_image_paths(input_root)
    logger.debug(f"检查输入目录得到的图像路径: {image_paths}")

    # 准备配置参数（仅包含可序列化的数据）
    config_params = {
        'feature_detector': args.feature_detector,
        'fusion_method': args.fusion_method,
        'tone_mapping': args.tone_mapping,
        'gamma': args.gamma,
        'saturation_scale': args.saturation_scale,
        'hue_shift': args.hue_shift,
        'dynamic_gamma': args.dynamic_gamma,
        'downscale_factor': args.downscale_factor,
        'noise_reduction': args.noise_reduction,  # Add noise reduction flag
    }


    # 记录所有使用的参数
    logger.info(f"配置参数: 特征检测算法={args.feature_detector}, 色调映射算法={args.tone_mapping}, "
                f"Gamma值={args.gamma}, 饱和度缩放比例={args.saturation_scale}, 色调偏移量={args.hue_shift} 度, "
                f"曝光融合方法={args.fusion_method}, 动态Gamma调整={'启用' if args.dynamic_gamma else '禁用'}, "
                f"下采样比例={args.downscale_factor}, 日志等级={args.log_level}")

    # 创建集中输出文件夹
    output_root = os.path.join(input_root, "output")
    os.makedirs(output_root, exist_ok=True)
    logger.info(f"集中输出文件夹: {output_root}")

    if image_paths:
        # 输入目录直接包含图像文件，作为单个处理集
        logger.info("输入目录直接包含图像文件，作为单个图像集进行处理。")
        subdir_name = os.path.basename(os.path.normpath(input_root))
        args_tuple = (input_root, subdir_name, output_root, config_params)
        process_subdirectory(args_tuple)
    else:
        # 输入目录不包含图像文件，假定其包含多个子目录
        logger.info("输入目录不直接包含图像文件，搜索子目录进行处理。")

        # 获取所有子文件夹，排除 'output' 文件夹
        subdirectories = get_subdirectories(input_root, exclude=['output'])
        logger.info(f"找到 {len(subdirectories)} 个子文件夹进行处理。")

        if not subdirectories:
            logger.error(f"主文件夹中未找到任何子文件夹: {input_root}")
            sys.exit(1)

        # 准备参数列表
        args_list = [
            (subdir_path, os.path.basename(subdir_path), output_root, config_params)
            for subdir_path in subdirectories
        ]

        # 使用多进程并行处理子文件夹
        logger.info(f"启动 {args.processes} 个进程进行并行处理。")
        try:
            with Pool(processes=args.processes) as pool:
                pool.map(process_subdirectory, args_list)
        except Exception as e:
            logger.exception(f"多进程处理时发生错误: {e}")

    logger.info("所有子文件夹的HDR多帧合成处理完成.")


if __name__ == "__main__":
    main()
