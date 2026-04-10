# Insta-360 INSV to Stereo VR Converter

This tool converts Insta-360 `.insv` files into a Stereo Top-Bottom (Bottom-Up) equirectangular format suitable for VR headsets and YouTube VR.

## Features
- **Stereo 1:1 Bottom-Up (WIP)**: Converts dual fisheye INSV to a stacked stereo format. *Note: Currently experimental and may have alignment issues.*
- **Hardware Acceleration**: Supports NVENC, VAAPI, and AMF.
- **System Monitoring**: Real-time CPU, RAM, and VRAM stats displayed during conversion.
- **YouTube VR Ready**: Automatically injects spatial media metadata for seamless VR playback on YouTube.
- **Professional CLI**: Progress tracking via `tqdm`.

## Requirements
- **FFmpeg**: Must be installed with `v360` filter support.
- **Python 3.8+**
- **Dependencies**:
  - `opencv-python`
  - `numpy`
  - `psutil`
  - `GPUtil`
  - `tqdm`

## Installation
```bash
pip install opencv-python numpy psutil GPUtil tqdm
```

## Usage
```bash
python convert.py input.insv output.mp4
```

## How it Works
1. **Extraction**: Extracts left and right fisheye streams from the INSV file.
2. **Projection**: Converts fisheye to equirectangular with a 1:1 aspect ratio per eye.
3. **Stacking**: Stacks the views in a "Bottom-Up" configuration (Right on top, Left on bottom).
4. **Encoding**: Encodes the final stream using HEVC for high quality and low file size.
5. **Metadata**: Uses the `spatial-media` tool to inject the required VR metadata tags.
