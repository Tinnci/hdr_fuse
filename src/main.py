# src/main.py

"""
主函数，协调各个组件完成HDR多帧合成处理流程（支持批处理和单个图像集处理）。
"""

import os
import argparse
import logging
import sys
from typing import List, Tuple

import cv2

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


def setup_logging():
    """
    设置全局日志配置。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler("hdr_pipeline.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

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
    获取指定文件夹中的所有图像文件路径。

    :param input_dir: 子文件夹路径
    :return: 图像文件路径列表
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
        and os.path.isfile(os.path.join(input_dir, f))
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


def process_subdirectory(
    subdir_path: str,
    subdir_name: str,
    output_root: str,
    reader: ImageReader,
    aligner: ImageAligner,
    hsv_processor: HSVProcessor,
    fusion: ExposureFusion,
    tone_mapper: ToneMapper,
    writer: ImageWriter,
    saturation_scale: float,
    hue_shift: float,
):
    """
    处理单个子文件夹中的图像集。

    :param subdir_path: 子文件夹路径
    :param subdir_name: 子文件夹名称
    :param output_root: 集中输出文件夹路径
    :param reader: ImageReader 实例
    :param aligner: ImageAligner 实例
    :param hsv_processor: HSVProcessor 实例
    :param fusion: ExposureFusion 实例
    :param tone_mapper: ToneMapper 实例
    :param writer: ImageWriter 实例
    :param saturation_scale: 饱和度缩放比例
    :param hue_shift: 色调偏移量（度）
    """
    logger = logging.getLogger("Main")
    logger.info(f"开始处理子文件夹: {subdir_path}")

    image_paths = get_image_paths(subdir_path)
    if not image_paths:
        logger.warning(f"子文件夹中未找到任何图像文件: {subdir_path}")
        return

    try:
        # 步骤 1: 读取图像
        logger.info(f"步骤 1: 读取图像 ({len(image_paths)} 张)。")
        images = reader.read_images(image_paths)
        logger.info(f"成功读取了 {len(images)} 张图像。")

        # 步骤 2: 对齐图像
        logger.info("步骤 2: 对齐图像。")
        aligned_images = aligner.align_images(images)
        logger.info("图像对齐完成。")

        # 步骤 3: 曝光融合
        logger.info("步骤 3: 曝光融合。")
        fused_image = fusion.fuse_images(aligned_images)
        logger.info("曝光融合完成。")

        # 步骤 4: 亮度和对比度增强（HSV处理）
        logger.info("步骤 4: 亮度和对比度增强。")
        hsv_image = hsv_processor.convert_to_hsv(fused_image)
        h, s, v = cv2.split(hsv_image)
        v = cv2.equalizeHist(v)  # 增强V通道
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_rgb = hsv_processor.convert_to_rgb(enhanced_hsv)
        logger.info("亮度和对比度增强完成。")

        # 步骤 5: 色调映射
        logger.info("步骤 5: 色调映射。")
        tone_mapped_image = tone_mapper.map_tone(enhanced_rgb)
        logger.info("色调映射完成。")

        # 步骤 6: 色彩调整（可选）
        if saturation_scale != 1.0 or hue_shift != 0.0:
            logger.info("步骤 6: 色彩调整。")
            hsv_adjusted = hsv_processor.convert_to_hsv(tone_mapped_image)
            if saturation_scale != 1.0:
                logger.debug(f"调整饱和度，缩放比例: {saturation_scale}")
                hsv_adjusted = hsv_processor.adjust_saturation(
                    hsv_adjusted, saturation_scale
                )
            if hue_shift != 0.0:
                logger.debug(f"调整色调，偏移量: {hue_shift} 度")
                hsv_adjusted = hsv_processor.adjust_hue(hsv_adjusted, hue_shift)
            tone_mapped_image = hsv_processor.convert_to_rgb(hsv_adjusted)
            logger.info("色彩调整完成。")

        # 生成输出路径
        output_in_subdir, output_in_output = generate_output_paths(
            subdir_path, subdir_name, output_root
        )

        # 步骤 7: 保存输出图像到子文件夹
        logger.info(f"步骤 7: 保存输出图像到子文件夹: {output_in_subdir}")
        writer.write_image(tone_mapped_image, output_in_subdir)

        # 步骤 8: 保存输出图像到集中输出文件夹
        logger.info(f"步骤 8: 保存输出图像到集中输出文件夹: {output_in_output}")
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


def main():
    """
    主函数，执行HDR多帧合成处理流程（支持批处理和单个图像集处理）。
    """
    setup_logging()
    logger = logging.getLogger("Main")
    args = parse_arguments()

    logger.info("HDR多帧合成处理系统启动。")

    input_root = args.input
    if not os.path.isdir(input_root):
        logger.error(f"输入路径不是一个有效的文件夹: {input_root}")
        sys.exit(1)

    # 检查输入目录是否直接包含图像文件
    logger.debug("检查输入目录是否包含图像文件。")
    image_paths = get_image_paths(input_root)

    if image_paths:
        # 输入目录直接包含图像文件，作为单个处理集
        logger.info("输入目录直接包含图像文件，作为单个图像集进行处理。")

        # 创建集中输出文件夹
        output_root = os.path.join(input_root, "output")
        os.makedirs(output_root, exist_ok=True)
        logger.info(f"集中输出文件夹: {output_root}")

        # 初始化组件
        logger.debug("初始化各个组件。")
        reader = ImageReader()
        aligner = ImageAligner(feature_detector=args.feature_detector)
        hsv_processor = HSVProcessor()
        fusion = ExposureFusion()
        tone_mapper = ToneMapper(method=args.tone_mapping, gamma=args.gamma)
        writer = ImageWriter()

        # 处理输入目录作为单个图像集
        subdir_name = os.path.basename(os.path.normpath(input_root))
        process_subdirectory(
            subdir_path=input_root,
            subdir_name=subdir_name,
            output_root=output_root,
            reader=reader,
            aligner=aligner,
            hsv_processor=hsv_processor,
            fusion=fusion,
            tone_mapper=tone_mapper,
            writer=writer,
            saturation_scale=args.saturation_scale,
            hue_shift=args.hue_shift,
        )
    else:
        # 输入目录不包含图像文件，假定其包含多个子目录
        logger.info("输入目录不直接包含图像文件，搜索子目录进行处理。")

        # 获取所有子文件夹，排除 'output' 文件夹
        subdirectories = get_subdirectories(input_root, exclude=['output'])
        logger.info(f"找到 {len(subdirectories)} 个子文件夹进行处理。")

        if not subdirectories:
            logger.error(f"主文件夹中未找到任何子文件夹: {input_root}")
            sys.exit(1)

        # 创建集中输出文件夹
        output_root = os.path.join(input_root, "output")
        os.makedirs(output_root, exist_ok=True)
        logger.info(f"集中输出文件夹: {output_root}")

        # 初始化组件
        logger.debug("初始化各个组件。")
        reader = ImageReader()
        aligner = ImageAligner(feature_detector=args.feature_detector)
        hsv_processor = HSVProcessor()
        fusion = ExposureFusion()
        tone_mapper = ToneMapper(method=args.tone_mapping, gamma=args.gamma)
        writer = ImageWriter()

        # 处理每个子文件夹
        for subdir_path in subdirectories:
            subdir_name = os.path.basename(subdir_path)
            process_subdirectory(
                subdir_path=subdir_path,
                subdir_name=subdir_name,
                output_root=output_root,
                reader=reader,
                aligner=aligner,
                hsv_processor=hsv_processor,
                fusion=fusion,
                tone_mapper=tone_mapper,
                writer=writer,
                saturation_scale=args.saturation_scale,
                hue_shift=args.hue_shift,
            )

    logger.info("所有子文件夹的HDR多帧合成处理完成。")


if __name__ == "__main__":
    main()
