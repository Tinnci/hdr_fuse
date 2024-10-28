# hdr_fuse

# Python HDR Multi-Frame Fusion Pipeline

![HDR](https://upload.wikimedia.org/wikipedia/commons/1/18/High_Dynamic_Range_Image_Example.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Graphical Interface](#graphical-interface)
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

Welcome to the **Python HDR Multi-Frame Fusion Pipeline**! This project automates the process of creating High Dynamic Range (HDR) images from multiple photographs taken at different exposure levels. By leveraging advanced image processing techniques and color space transformations, this pipeline ensures high-quality HDR image synthesis with optimal brightness and minimal color distortion.

## Features

- **Graphical User Interface**: Includes a PyQt6 GUI for user-friendly operation.
- **Batch Processing**: Process multiple sets of images organized in subfolders.
- **Flexible Input**: Accepts both single images and directories with multiple exposure sets.
- **Image Alignment**: Uses SIFT or ORB for precise image alignment.
- **Exposure Fusion**: Fuses multiple exposures to create a single HDR image.
- **Brightness and Contrast Enhancement**: Adjusts brightness and contrast in HSV color space.
- **Tone Mapping**: Converts HDR to LDR using advanced tone mapping algorithms (Reinhard, Drago, Durand).
- **Color Adjustment**: Adjusts saturation and hue for improved color fidelity.
- **Detailed Logging**: Logs every step for easy debugging.
- **Error Handling**: Manages exceptions to ensure smooth batch processing.

## Directory Structure

```
hdr_fuse/
├── gui/
│   ├── main_window.py              # GUI implementation with PyQt6
├── src/
│   ├── __init__.py                  # Module initializer
│   ├── main.py                      # Main execution script for command-line interface
│   ├── image_reader.py              # Image reading component
│   ├── image_aligner.py             # Image alignment component
│   ├── exposure_fusion.py           # Exposure fusion component
│   ├── tone_mapping.py              # Tone mapping component
│   ├── hsv_processing.py            # HSV color space processing component
│   ├── image_writer.py              # Image saving component
│   └── exceptions.py                # Custom exception classes
├── tests/                           # Unit tests
├── .gitignore
├── .pdm-python
├── pyproject.toml                   # Project configuration
├── pdm.lock                         # Dependency lock file
└── README.md                        # Project documentation
```

## Installation

### Prerequisites

- **Python 3.7 or higher**: Install from [Python's official website](https://www.python.org/downloads/).
- **PDM**: Use PDM for dependency management. Install via pip:

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

If you prefer `pip`, activate a virtual environment and run:

```bash
pip install -r requirements.txt
```

*Note: Ensure `requirements.txt` includes all required packages.*

## Usage

### Graphical Interface

A graphical interface, built with PyQt6, provides an intuitive way to configure and run HDR processing:

1. **Launch GUI**:
   ```bash
   python gui/main_window.py
   ```
2. **Interface Options**:
   - **Input & Output**: Select input folder and output destination.
   - **Feature Detection**: Choose SIFT or ORB for image alignment.
   - **Tone Mapping**: Select tone mapping algorithm (Reinhard, Drago, or Durand).
   - **Adjustments**: Set gamma, saturation scale, hue shift, and enable dynamic gamma or noise reduction.
   - **Start/Cancel**: Begin or cancel HDR processing.
   - **Progress & Log**: Monitor processing progress and log output in real time.

### Single Folder Processing

To process a single set of images:

```bash
python src/main.py -i "path/to/your/image_set_folder" -f ORB -t Drago --gamma 2.2 --saturation_scale 1.2 --hue_shift 10
```

Parameters:
- `-i`, `--input`: Path to input folder with images at different exposures.
- `-f`, `--feature_detector`: Feature detection algorithm (SIFT or ORB).
- `-t`, `--tone_mapping`: Tone mapping algorithm (Reinhard, Drago, Durand).
- `--gamma`: Gamma correction value.
- `--saturation_scale`: Saturation adjustment factor.
- `--hue_shift`: Hue shift in degrees.

Output:
- Fused HDR image saved in input folder as `{folder_name}_fused.jpg`.
- Centralized output in `output` folder as `{folder_name}.jpg`.

### Batch Processing

For batch processing of images in subfolders:

```bash
python src/main.py -i "D:\hdr" -f ORB -t Drago --gamma 2.2 --saturation_scale 1.2 --hue_shift 10
```

Example Directory Structure:

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
└── output\
    ├── 1.jpg
    ├── 2.jpg
    └── 50.jpg
```

Output:
- Each subfolder contains a fused image (`{subfolder_name}_fused.jpg`).
- Centralized output in the main `output` folder.

## Command-Line Arguments

| Argument             | Description                                 | Required | Default    |
|----------------------|---------------------------------------------|----------|------------|
| `-i`, `--input`      | Input image folder or directory             | Yes      | N/A        |
| `-f`, `--feature_detector` | Feature detection algorithm (SIFT, ORB) | No       | `SIFT`     |
| `-t`, `--tone_mapping` | Tone mapping algorithm (Reinhard, Drago, Durand) | No | `Reinhard` |
| `--gamma`            | Gamma correction                           | No       | `1.0`      |
| `--saturation_scale` | Saturation adjustment factor               | No       | `1.0`      |
| `--hue_shift`        | Hue shift in degrees                       | No       | `0.0`      |

## Logging

Logs are created for tracking each step, saved in `hdr_pipeline.log`.

- **Console Logs**: Basic info and error warnings.
- **Log File**: Detailed logs with debug information, located in `hdr_pipeline.log`.

## Error Handling

The pipeline handles exceptions gracefully, ensuring uninterrupted batch processing:

- **Custom Exceptions**: Specific errors are managed via custom exceptions.
- **Logging Errors**: All exceptions are logged with detailed messages.

Common Errors:
- **Missing Dependencies**: Ensure dependencies are installed.
- **Invalid Input Paths**: Verify input paths and valid image files.
- **Unsupported Formats**: The pipeline supports `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`.

## Dependencies

This project requires:

- **[Pillow](https://python-pillow.org/)**: Image handling.
- **[OpenCV-Python](https://opencv.org/)**: Image processing and computer vision.
- **[NumPy](https://numpy.org/)**: Numerical operations.
- **[Logging](https://docs.python.org/3/library/logging.html)**: Logging.

Install dependencies with:

```bash
pdm install
```

Or with pip:

```bash
pip install Pillow opencv-python numpy
```

## Performance Optimization

Optimize for large batches or high-resolution images:

- **Multi-Processing**: Use Python’s `multiprocessing` for parallelization.
- **GPU Acceleration**: Utilize CUDA with OpenCV if available.
- **Memory Management**: Batch process and clear memory after each set.

## Extensibility

The pipeline is designed for easy expansion:

- **New Tone Mapping Algorithms**: Extend the `ToneMapper` class.
- **Additional Image Formats**: Modify `get_image_paths` for more formats.
- **Enhanced Alignment**: Add advanced alignment techniques.
- **GUI Customization**: Update the PyQt6 GUI for additional features.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or contributions, please reach out via the project repository.

---

*Happy HDR Imaging! 🚀*