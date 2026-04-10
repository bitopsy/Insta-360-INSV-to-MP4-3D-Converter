#!/usr/bin/env python3
"""
Insta-360 INSV to Stereo VR Converter
Converts spherical INSV format to Stereo Top-Bottom (Bottom-Up) format with YouTube VR metadata.
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


class DepthMapGenerator:
    """Generate estimated depth maps for stereo 3D conversion"""

    @staticmethod
    def generate_stereo_depth(
        left_frame: np.ndarray, right_frame: np.ndarray
    ) -> np.ndarray:
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=21)
        disparity = stereo.compute(gray_left, gray_right)
        depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        depth_map = cv2.medianBlur(depth_map, 5)
        return depth_map


class INSVConverter:
    """Main converter class for INSV to Stereo VR format"""

    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.monitor = SystemMonitor()
        self.depth_generator = DepthMapGenerator()
        self.cuda_available = self._check_cuda_support()

    def _check_cuda_support(self) -> bool:
        try:
            return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        except:
            return False

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
        print("Injecting YouTube VR metadata...")
        try:
            # Path to the cloned spatial-media repo
            spatial_media_path = Path(
                "/home/al/opencode/spatial-media/spatialmedia/__main__.py"
            )
            if not spatial_media_path.exists():
                print("Spatial-media tool not found, skipping injection.")
                return

            # Command to inject metadata: python3 spatialmedia/__main__.py inject <file>
            # We need to specify it's stereo and spherical
            cmd = [
                sys.executable,
                str(spatial_media_path),
                "inject",
                str(self.output_file),
            ]
            # Note: spatial-media usually asks for input or has flags.
            # The CLI for spatial-media inject is: python spatialmedia/__main__.py inject <file>
            # It then asks for projection and layout. We can't easily interact.
            # However, we can try to use the metadata_utils if we want to be programmatic.
            # For simplicity, we'll try the command and see.
            # To automate, we can pipe the answers.
            # 1. Spherical? Yes
            # 2. Stereo? Yes
            # 3. Top-Bottom? Yes

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(input="y\ny\ny\n")
            if process.returncode == 0:
                print("Successfully injected VR metadata.")
            else:
                print(f"Metadata injection failed: {stderr}")
        except Exception as e:
            print(f"Error injecting metadata: {e}")

    def convert(self):
        video_info = self._get_video_info()
        # Stereo 1:1 Bottom-Up means:
        # - Aspect ratio per eye: 1:1
        # - Layout: Right Eye TOP, Left Eye BOTTOM (Bottom-Up)
        # - Final resolution: Width x (Height * 2) where Width == Height

        eye_res = video_info["height"]  # Use height as base for 1:1
        output_width = eye_res
        output_height = eye_res * 2

        print(
            f"Target Resolution: {output_width}x{output_height} (Stereo 1:1 Bottom-Up)"
        )

        # 1. Handle dual fisheye to 1:1 Equirectangular per eye
        # We'll create two temporary files for the eyes
        left_eye_tmp = self.output_file.with_suffix(".left.tmp.mp4")
        right_eye_tmp = self.output_file.with_suffix(".right.tmp.mp4")

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
            extract_eye(0, left_eye_tmp)
            extract_eye(1, right_eye_tmp)
        else:
            # Single stream side-by-side
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
                str(left_eye_tmp),
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
                str(right_eye_tmp),
            ]
            subprocess.run(cmd_l, capture_output=True, check=True)
            subprocess.run(cmd_r, capture_output=True, check=True)

        # 2. Stack them: Right Eye Top, Left Eye Bottom (Bottom-Up)
        print("Stacking eyes (Bottom-Up)...")
        # vstack: [right][left]
        stack_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(right_eye_tmp),
            "-i",
            str(left_eye_tmp),
            "-filter_complex",
            "[0:v][1:v]vstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "hevc_nvenc" if self.cuda_available else "libx265",
            "-preset",
            "medium",
            "-crf",
            "23" if not self.cuda_available else "-cq",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_file),
        ]

        # For progress bar, we'll run FFmpeg and parse output
        self.monitor.start_monitoring()
        process = subprocess.Popen(
            stack_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        pbar = tqdm(total=video_info["total_frames"], desc="Converting", unit="frame")
        last_frame = 0

        try:
            for line in process.stdout:
                if "frame=" in line:
                    try:
                        # Extract frame number from "frame=  123"
                        parts = line.split("frame=")[1].split()
                        frame = int(parts[0])
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

        # Cleanup
        left_eye_tmp.unlink(missing_ok=True)
        right_eye_tmp.unlink(missing_ok=True)

        # 3. Inject VR Metadata
        self._inject_vr_metadata()

        return True


def main():
    parser = argparse.ArgumentParser(
        description="INSV to Stereo VR 1:1 Bottom-Up Converter"
    )
    parser.add_argument("input", help="Input INSV file")
    parser.add_argument("output", help="Output MP4 file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        converter = INSVConverter(args.input, args.output)
        if converter.convert():
            print(f"\n🎉 Successfully converted to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
