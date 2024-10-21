# src/exceptions.py

"""
定义自定义异常类，用于HDR多帧合成处理系统中的错误处理。
"""

class ImageReadError(Exception):
    """在图像读取过程中出现问题时抛出，例如文件不存在、格式不支持等。"""
    pass

class ImageAlignError(Exception):
    """图像对齐失败时抛出，例如特征点检测或匹配失败、计算变换矩阵错误等。"""
    pass

class ExposureFusionError(Exception):
    """曝光融合过程中出现问题时抛出，例如数组操作错误、权重计算错误等。"""
    pass

class ToneMappingError(Exception):
    """色调映射失败时抛出，例如色调映射算法错误、参数设置错误等。"""
    pass

class ImageWriteError(Exception):
    """图像保存失败时抛出，例如文件写入错误、路径无效等。"""
    pass
