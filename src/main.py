"""
主函数，协调各个组件完成HDR多帧合成处理流程。
"""

import os
import argparse
import logging
import sys
from venv import logger
import cv2
from hdr_fuse.image_reader import ImageReader
from hdr_fuse.image_aligner import ImageAligner
from hdr_fuse.exposure_fusion import ExposureFusion
from hdr_fuse.tone_mapping import ToneMapper
from hdr_fuse.image_writer import ImageWriter
from hdr_fuse.hsv_processing import HSVProcessor
from hdr_fuse.exceptions import ImageReadError, ImageAlignError, ExposureFusionError, ToneMappingError, ImageWriteError

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
    fh = logging.FileHandler('hdr_pipeline.log', mode='w')
    fh.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(ch)
    logger.addHandler(fh)

def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Python HDR 多帧合成管线")
    parser.add_argument('-i', '--input', required=True, help='输入图像文件路径或文件夹')
    parser.add_argument('-f', '--feature_detector', choices=['SIFT', 'ORB'], default='SIFT', help='特征点检测算法')
    parser.add_argument('-t', '--tone_mapping', choices=['Reinhard', 'Drago', 'Durand'], default='Reinhard', help='色调映射算法')
    parser.add_argument('--gamma', type=float, default=1.0, help='色调映射gamma值')
    parser.add_argument('--saturation_scale', type=float, default=1.0, help='饱和度缩放比例')
    parser.add_argument('--hue_shift', type=float, default=0.0, help='色调偏移量（度）')
    return parser.parse_args()

def get_image_paths(input_path):
    """
    如果输入是文件夹，获取该文件夹中的所有图像文件路径。否则返回输入文件的路径。
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    if os.path.isdir(input_path):
        logger.info(f"输入是文件夹，读取文件夹中的图像: {input_path}")
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]
        if not image_paths:
            logger.error(f"文件夹中未找到图像文件: {input_path}")
            sys.exit(1)
        return image_paths
    elif os.path.isfile(input_path) and input_path.lower().endswith(valid_extensions):
        logger.info(f"输入是单个图像文件: {input_path}")
        return [input_path]
    else:
        logger.error(f"输入路径无效或不包含有效的图像文件: {input_path}")
        sys.exit(1)

def generate_output_path(input_file):
    """
    根据输入文件名生成输出文件名，并将其保存在同一文件夹内，文件名格式为 inputname_fused.jpg。
    """
    base, ext = os.path.splitext(input_file)
    return f"{base}_fused.jpg"

def main():
    """
    主函数，执行HDR多帧合成处理流程。
    """
    setup_logging()
    logger = logging.getLogger('Main')
    args = parse_arguments()

    logger.info("HDR多帧合成处理系统启动。")

    # 获取所有图像路径
    image_paths = get_image_paths(args.input)
    logger.info(f"读取了 {len(image_paths)} 个输入文件。")

    # 初始化组件
    logger.debug("初始化各个组件。")
    reader = ImageReader()
    aligner = ImageAligner(feature_detector=args.feature_detector)
    hsv_processor = HSVProcessor()
    fusion = ExposureFusion()
    tone_mapper = ToneMapper(method=args.tone_mapping, gamma=args.gamma)
    writer = ImageWriter()

    try:
        # 步骤 1: 读取图像
        logger.info("步骤 1: 读取图像。")
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
        if args.saturation_scale != 1.0 or args.hue_shift != 0.0:
            logger.info("步骤 6: 色彩调整。")
            hsv_adjusted = hsv_processor.convert_to_hsv(tone_mapped_image)
            if args.saturation_scale != 1.0:
                logger.debug(f"调整饱和度，缩放比例: {args.saturation_scale}")
                hsv_adjusted = hsv_processor.adjust_saturation(hsv_adjusted, args.saturation_scale)
            if args.hue_shift != 0.0:
                logger.debug(f"调整色调，偏移量: {args.hue_shift} 度")
                hsv_adjusted = hsv_processor.adjust_hue(hsv_adjusted, args.hue_shift)
            tone_mapped_image = hsv_processor.convert_to_rgb(hsv_adjusted)
            logger.info("色彩调整完成。")

        # 步骤 7: 保存输出图像
        logger.info("步骤 7: 保存输出图像。")
        output_path = generate_output_path(image_paths[0])  # 根据输入图像生成输出路径
        writer.write_image(tone_mapped_image, output_path)
        logger.info(f"图像成功保存到 {output_path}。")

    except ImageReadError as e:
        logger.error(f"图像读取错误: {e}")
        sys.exit(1)
    except ImageAlignError as e:
        logger.error(f"图像对齐错误: {e}")
        sys.exit(1)
    except ExposureFusionError as e:
        logger.error(f"曝光融合错误: {e}")
        sys.exit(1)
    except ToneMappingError as e:
        logger.error(f"色调映射错误: {e}")
        sys.exit(1)
    except ImageWriteError as e:
        logger.error(f"图像保存错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"发生未处理的错误: {e}")
        sys.exit(1)

    logger.info("HDR多帧合成处理系统完成。")

if __name__ == "__main__":
    main()
