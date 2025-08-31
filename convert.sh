#!/bin/bash
# Input and output file paths
INPUT_FILE="/path/to/file.insv"
OUTPUT_FILE="${INPUT_FILE%.*}_equirectangular.mp4"

# First, probe the file to verify streams and get detailed info
echo "Probing input file..."
ffprobe -v quiet -print_format json -show_streams -show_format "$INPUT_FILE"

echo ""
echo "Converting INSV to equirectangular format..."

# Convert INSV to equirectangular 2:1 format
# First check available v360 formats, then use basic fisheye conversion
echo "Checking available v360 formats..."
ffmpeg -hide_banner -f lavfi -i nullsrc -vf v360=help -f null - 2>&1 | grep -A 20 "input format"

echo ""
echo "Converting using basic fisheye format..."
ffmpeg -i "$INPUT_FILE" \
    -vf "v360=fisheye:equirect:ih_fov=190:iv_fov=190" \
    -c:v hevc_nvenc \
    -preset medium \
    -b:v 50M \
    -c:a aac \
    -b:a 192k \
    -movflags +faststart \
    "$OUTPUT_FILE"

echo "Conversion complete: $OUTPUT_FILE"

# Probe the output file to verify the conversion
echo ""
echo "Probing output file..."
ffprobe -v quiet -print_format json -show_streams -show_format "$OUTPUT_FILE"
