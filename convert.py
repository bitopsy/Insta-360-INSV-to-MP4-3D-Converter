#!/usr/bin/env python3
"""
Insta-360 INSV to VR Converter
Professional CLI for converting INSV to VR formats (Mono/Stereo) with HW acceleration.
"""

import os
import sys
import time
import threading
import subprocess
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
import psutil
import GPUtil
import cv2
import numpy as np
from tqdm import tqdm
from datetime import timedelta


class SystemMonitor:
    """Monitor CPU, memory, and VRAM usage"""

    def __init__(self):
        self.monitoring = False
        self.current_stats = {"cpu": 0, "memory": 0, "vram": 0, "vram_mb": 0}

    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory().percent
                vram_p, vram_m = 0, 0
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    vram_p = gpu.memoryUtil * 100
                    vram_m = gpu.memoryUsed
                self.current_stats = {
                    "cpu": cpu,
                    "memory": mem,
                    "vram": vram_p,
                    "vram_mb": vram_m,
                }
            except:
                pass
            time.sleep(1)

    def get_stats_str(self) -> str:
        s = self.current_stats
        return f"CPU: {s['cpu']:.1f}% | RAM: {s['memory']:.1f}% | VRAM: {s['vram']:.1f}% ({s['vram_mb']:.0f}MB)"


class INSVConverter:
    """Main converter class for INSV to VR format"""

    ENCODERS = {
        "nvenc": {"hevc": "hevc_nvenc", "h264": "h264_nvenc"},
        "vaapi": {"hevc": "hevc_vaapi", "h264": "h264_vaapi"},
        "amf": {"hevc": "hevc_amf", "h264": "h264_amf"},
        "software": {"hevc": "libx265", "h264": "libx264"},
    }

    def __init__(
        self,
        input_file: str,
        output_file: Optional[str],
        stereo: bool,
        encoder: str,
        codec: str,
    ):
        self.input_file = Path(input_file)
        self.stereo = stereo
        self.encoder_type = encoder
        self.codec_type = codec
        self.monitor = SystemMonitor()

        if output_file:
            self.output_file = Path(output_file)
        else:
            self.output_file = self.input_file.with_suffix(".mp4")

    def _get_video_info(self) -> dict:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(self.input_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        video_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
        if not video_streams:
            raise ValueError("No video streams found")

        vs = video_streams[0]
        fps_str = vs.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = (
            float(fps_parts[0]) / float(fps_parts[1])
            if len(fps_parts) == 2
            else float(fps_parts[0])
        )
        duration = float(info["format"].get("duration", 0))

        return {
            "duration": duration,
            "width": int(vs["width"]),
            "height": int(vs["height"]),
            "fps": fps,
            "total_frames": int(duration * fps) if duration > 0 else 0,
            "stream_count": len(video_streams),
        }

    def _inject_vr_metadata(self):
        """Inject YouTube VR metadata using spatial-media tool"""
        if not self.stereo:
            return

        print("\nInjecting YouTube VR metadata...")
        try:
            # Look for spatial-media tool in expected location
            spatial_media_path = Path(
                "/home/al/opencode/spatial-media/spatialmedia/__main__.py"
            )
            if not spatial_media_path.exists():
                print("Spatial-media tool not found, skipping injection.")
                return

            if not self.output_file.exists():
                print(f"Output file {self.output_file} not found, skipping injection.")
                return

            cmd = [
                sys.executable,
                str(spatial_media_path),
                "inject",
                str(self.output_file),
            ]
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Answer: Spherical? Yes, Stereo? Yes, Top-Bottom? Yes
            stdout, stderr = process.communicate(input="y\ny\ny\n")
            if process.returncode == 0:
                print("Successfully injected VR metadata.")
            else:
                print(f"Metadata injection failed: {stderr}")
        except Exception as e:
            print(f"Error injecting metadata: {e}")

    def convert(self):
        video_info = self._get_video_info()
        encoder_name = self.ENCODERS[self.encoder_type][self.codec_type]

        if self.stereo:
            # Stereo 1:1 Bottom-Up (Right Top, Left Bottom)
            eye_res = video_info["height"]
            output_width, output_height = eye_res, eye_res * 2

            print(f"Mode: Stereo Bottom-Up | Res: {output_width}x{output_height}")

            left_tmp = self.output_file.with_suffix(".left.tmp.mp4")
            right_tmp = self.output_file.with_suffix(".right.tmp.mp4")

            def extract_eye(stream_idx, out_path):
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.input_file),
                    "-map",
                    f"0:v:{stream_idx}",
                    "-vf",
                    f"v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width}:{eye_res},setsar=1",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_path),
                ]
                subprocess.run(cmd, capture_output=True, check=True)

            print("Extracting eyes...")
            if video_info["stream_count"] >= 2:
                extract_eye(0, left_tmp)
                extract_eye(1, right_tmp)
            else:
                cmd_l = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.input_file),
                    "-vf",
                    f"crop=iw/2:ih:0:0,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width}:{eye_res},setsar=1",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(left_tmp),
                ]
                cmd_r = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.input_file),
                    "-vf",
                    f"crop=iw/2:ih:iw/2:0,v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width}:{eye_res},setsar=1",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(right_tmp),
                ]
                subprocess.run(cmd_l, capture_output=True, check=True)
                subprocess.run(cmd_r, capture_output=True, check=True)

            # Stack: Right Top, Left Bottom
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(right_tmp),
                "-i",
                str(left_tmp),
                "-filter_complex",
                "[0:v][1:v]vstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                encoder_name,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                str(self.output_file),
            ]

            # Cleanup temporary files after conversion
            cleanup_files = [left_tmp, right_tmp]
        else:
            # Mono Equirectangular 2:1
            output_width = video_info["width"]  # Keep original width
            output_height = output_width // 2
            print(f"Mode: Mono Equirectangular | Res: {output_width}x{output_height}")

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_file),
                "-vf",
                f"v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width}:{output_height},setsar=1",
                "-c:v",
                encoder_name,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                str(self.output_file),
            ]
            cleanup_files = []

        # Add encoder presets
        if "nvenc" in encoder_name:
            cmd.insert(-1, "-cq")
            cmd.insert(-1, "23")
        elif "lib" in encoder_name:
            cmd.insert(-1, "-crf")
            cmd.insert(-1, "23")

        self.monitor.start_monitoring()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        pbar = tqdm(total=video_info["total_frames"], desc="Processing", unit="frame")
        last_frame = 0

        try:
            for line in process.stdout:
                if "frame=" in line:
                    try:
                        frame = int(line.split("frame=")[1].split()[0])
                        if frame > last_frame:
                            pbar.update(frame - last_frame)
                            last_frame = frame
                            pbar.set_postfix_str(self.monitor.get_stats_str())
                    except:
                        pass
            process.wait()
            pbar.close()
        finally:
            self.monitor.stop_monitoring()
            for f in cleanup_files:
                f.unlink(missing_ok=True)

        self._inject_vr_metadata()
        return True


def main():
    parser = argparse.ArgumentParser(description="Insta-360 INSV to VR Converter")
    parser.add_argument("input", help="Input INSV file")
    parser.add_argument(
        "-o", "--output", help="Output MP4 file (default: input_name.mp4)"
    )
    parser.add_argument(
        "--stereo", action="store_true", help="Enable stereo 1:1 Bottom-Up conversion"
    )
    parser.add_argument(
        "--encoder",
        choices=["nvenc", "vaapi", "amf", "software"],
        default="software",
        help="Hardware encoder to use",
    )
    parser.add_argument(
        "--codec", choices=["hevc", "h264"], default="hevc", help="Video codec to use"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        converter = INSVConverter(
            args.input, args.output, args.stereo, args.encoder, args.codec
        )
        if converter.convert():
            print(f"\n🎉 Successfully converted to {converter.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
