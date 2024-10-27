# hdr_fuse

# Python HDR Multi-Frame Fusion Pipeline

![HDR](https://upload.wikimedia.org/wikipedia/commons/1/18/High_Dynamic_Range_Image_Example.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Single Folder Processing](#single-folder-processing)
  - [Batch Processing](#batch-processing)
- [Command-Line Arguments](#command-line-arguments)
- [Logging](#logging)
- [Error Handling](#error-handling)
- [Dependencies](#dependencies)
- [Performance Optimization](#performance-optimization)
- [Extensibility](#extensibility)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **Python HDR Multi-Frame Fusion Pipeline**! This project is designed to automate the process of creating High Dynamic Range (HDR) images from multiple photographs taken at different exposure levels. By leveraging advanced image processing techniques and color space transformations, this pipeline ensures high-quality HDR image synthesis with minimal color distortion and optimal brightness.

## Features

- **Batch Processing**: Automatically process multiple sets of images organized in subfolders.
- **Flexible Input**: Accepts individual image files or entire directories containing multiple exposure sets.
- **Image Alignment**: Utilizes feature detection (SIFT or ORB) to align images precisely.
- **Exposure Fusion**: Combines multiple exposures to create a single HDR image using HSV color space.
- **Brightness and Contrast Enhancement**: Enhances the V channel in HSV for optimal brightness and contrast.
- **Tone Mapping**: Applies advanced tone mapping algorithms (Reinhard, Drago, Durand) to convert HDR to LDR.
- **Color Adjustment**: Optionally adjusts saturation and hue for enhanced color fidelity.
- **Detailed Logging**: Comprehensive logs for monitoring processing steps and debugging.
- **Robust Error Handling**: Gracefully handles exceptions to ensure uninterrupted batch processing.

## Directory Structure

```
hdr_fuse/
├── src/
│   ├── __init__.py                  # Module initializer
│   ├── image_reader.py              # Image reading component
│   ├── image_aligner.py             # Image alignment component
│   ├── exposure_fusion.py           # Exposure fusion component
│   ├── tone_mapping.py              # Tone mapping component
│   ├── hsv_processing.py            # HSV space processing component
│   ├── image_writer.py              # Image writing component
│   └── exceptions.py                # Custom exception classes
├── tests/                           # Unit tests
├── .gitignore
├── .pdm-python
├── main.py                          # Main execution script
├── pyproject.toml                   # Project configuration
├── pdm.lock                         # Dependency lock file
├── README.md                        # Project documentation
└── hdr_pipeline.log                  # Log file (generated after execution)
```

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure that Python is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
- **PDM (Python Development Master)**: Used for dependency management. Install via pip:

  ```bash
  pip install pdm
  ```

### Clone the Repository

```bash
git clone https://github.com/tinnci/hdr_fuse.git
cd hdr_fuse
```

### Install Dependencies

Use PDM to install the required dependencies:

```bash
pdm install
```

Alternatively, if you prefer using `pip`, ensure you have a virtual environment activated and run:

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` file should list all necessary packages if you choose this method.*

## Usage

The main script `main.py` supports both single-folder and batch processing of HDR image sets.

### Single Folder Processing

Process a single set of images within a folder.

**Command:**

```bash
python main.py -i "path/to/your/image_set_folder" -f ORB -t Drago --gamma 2.2 --saturation_scale 1.2 --hue_shift 10
```

**Parameters:**

- `-i` or `--input`: Path to the input folder containing images with different exposures.
- `-f` or `--feature_detector`: Feature detection algorithm (`SIFT` or `ORB`).
- `-t` or `--tone_mapping`: Tone mapping algorithm (`Reinhard`, `Drago`, `Durand`).
- `--gamma`: Gamma correction value for tone mapping.
- `--saturation_scale`: Scale factor for saturation adjustment.
- `--hue_shift`: Degree shift for hue adjustment.

**Output:**

- The fused HDR image is saved in the input folder with the name `{folder_name}_fused.jpg`.
- A centralized output image is saved in the `output` subfolder within the input directory as `{folder_name}.jpg`.

### Batch Processing

Process multiple sets of images organized in subfolders.

**Directory Structure Example:**

```
D:\hdr\
├── 1\
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── 2\
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
...
└── 50\
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

**Command:**

```bash
python main.py -i "D:\hdr" -f ORB -t Drago --gamma 2.2 --saturation_scale 1.2 --hue_shift 10
```

**Parameters:**

- `-i` or `--input`: Path to the main input directory containing multiple subdirectories, each with a set of images.
- `-f` or `--feature_detector`: Feature detection algorithm (`SIFT` or `ORB`).
- `-t` or `--tone_mapping`: Tone mapping algorithm (`Reinhard`, `Drago`, `Durand`).
- `--gamma`: Gamma correction value for tone mapping.
- `--saturation_scale`: Scale factor for saturation adjustment.
- `--hue_shift`: Degree shift for hue adjustment.

**Output:**

- **Per Subfolder:**
  - Each subfolder will contain a fused image named `{subfolder_name}_fused.jpg`.
- **Centralized Output Folder:**
  - An `output` folder is created within the main input directory.
  - Each fused image is also saved here with the name `{subfolder_name}.jpg`.

**Example:**

After processing, your directory might look like:

```
D:\hdr\
├── 1\
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── 1_fused.jpg
├── 2\
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── 2_fused.jpg
...
├── 50\
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── 50_fused.jpg
└── output\
    ├── 1.jpg
    ├── 2.jpg
    ...
    └── 50.jpg
```

## Command-Line Arguments

| Argument             | Description                                                                                         | Required | Default   |
|----------------------|-----------------------------------------------------------------------------------------------------|----------|-----------|
| `-i`, `--input`      | Path to the input image file or directory. For batch processing, specify the main directory containing subfolders. | Yes      | N/A       |
| `-f`, `--feature_detector` | Feature detection algorithm to use (`SIFT` or `ORB`).                                        | No       | `SIFT`    |
| `-t`, `--tone_mapping`      | Tone mapping algorithm to use (`Reinhard`, `Drago`, `Durand`).                              | No       | `Reinhard`|
| `--gamma`            | Gamma correction value for tone mapping.                                                           | No       | `1.0`     |
| `--saturation_scale` | Scale factor for saturation adjustment.                                                             | No       | `1.0`     |
| `--hue_shift`        | Degree shift for hue adjustment.                                                                    | No       | `0.0`     |

## Logging

The pipeline employs Python's `logging` module to provide detailed insights into the processing steps. Logs are outputted both to the console and a log file named `hdr_pipeline.log`.

- **Console Logs**: Display essential information and warnings/errors at the `INFO` level.
- **Log File**: Contains comprehensive debug information, including step-by-step processing details, saved in `hdr_pipeline.log` in the script's execution directory.

**Example Log Entries:**

```
2024-10-21 16:30:00,123 - Main - INFO - HDR多帧合成处理系统启动。
2024-10-21 16:30:00,456 - Main - INFO - 开始处理子文件夹: D:\hdr\1
2024-10-21 16:30:00,789 - Main - INFO - 步骤 1: 读取图像。
2024-10-21 16:30:01,012 - Main - DEBUG - 读取图像: D:\hdr\1\img1.jpg
...
2024-10-21 16:30:05,678 - Main - INFO - 图像成功保存到 D:\hdr\1\1_fused.jpg。
2024-10-21 16:30:05,679 - Main - INFO - 成功处理子文件夹: D:\hdr\1
...
```

## Error Handling

The pipeline is equipped with robust error handling mechanisms to ensure smooth batch processing:

- **Custom Exceptions**: Defined in `src/exceptions.py` to handle specific error scenarios.
- **Graceful Failures**: If an error occurs while processing a subfolder, the pipeline logs the error and continues processing the remaining subfolders.
- **Logging Errors**: All exceptions are logged with detailed messages to facilitate debugging.

**Common Errors:**

- **Missing Imports**: Ensure all dependencies are installed and the `PYTHONPATH` is correctly set.
- **Invalid Input Paths**: Verify that the input paths are correct and contain valid image files.
- **Unsupported File Formats**: The pipeline supports `.jpg`, `.jpeg`, `.png`, `.tiff`, and `.bmp`. Other formats may cause errors.

## Dependencies

The project relies on several Python libraries for image processing and system operations:

- **[Pillow](https://python-pillow.org/)**: Image handling and manipulation.
- **[OpenCV-Python](https://opencv.org/)**: Advanced computer vision and image processing.
- **[NumPy](https://numpy.org/)**: Numerical operations on image data.
- **[Logging](https://docs.python.org/3/library/logging.html)**: Logging framework for diagnostics.

**Installation:**

Ensure you have the necessary dependencies installed. Using PDM:

```bash
pdm install
```

Or using pip:

```bash
pip install Pillow opencv-python numpy
```

## Performance Optimization

Processing large batches of high-resolution images can be resource-intensive. Consider the following optimizations:

- **Multi-Threading/Processing**: Utilize Python's `concurrent.futures` or `multiprocessing` modules to parallelize processing across multiple CPU cores.
- **GPU Acceleration**: Leverage OpenCV's CUDA modules for faster image processing if a compatible GPU is available.
- **Memory Management**: Process images in batches and release memory after processing each set to prevent memory leaks.

## Extensibility

The pipeline is designed with modularity in mind, allowing for easy extension and customization:

- **Additional Tone Mapping Algorithms**: Implement new tone mapping techniques by extending the `ToneMapper` class.
- **Support for More Image Formats**: Modify the `get_image_paths` function to include additional file extensions.
- **Enhanced Image Alignment**: Incorporate more advanced alignment techniques or improve existing methods.
- **GUI Integration**: Develop a graphical user interface for more user-friendly operation.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, suggestions, or contributions, please contact:

- **Your Name**
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

---

*Happy HDR Imaging! 🚀*

```