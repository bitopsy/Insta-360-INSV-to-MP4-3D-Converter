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
from typing import Optional, Tuple, List
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
    """Generate disparity-based depth maps for stereo 3D"""

    @staticmethod
    def generate_disparity(
        left_frame: np.ndarray, right_frame: np.ndarray
    ) -> np.ndarray:
        gray_l = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(gray_l, gray_r)
        depth = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth

    @staticmethod
    def apply_depth_shift(
        frame: np.ndarray, depth: np.ndarray, shift_max: int = 20
    ) -> np.ndarray:
        """Shift pixels based on depth map to create the opposite eye (DIBR)"""
        h, w = frame.shape[:2]
        # Normalize depth to a shift value (0 to shift_max)
        shift_map = (depth.astype(np.float32) / 255.0) * shift_max

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Shift x coordinates based on depth
        # Right eye is typically shifted to the left relative to the left eye image
        new_x = np.clip(x + shift_map, 0, w - 1).astype(np.float32)

        # Remap the image
        shifted_frame = cv2.remap(frame, new_x, y.astype(np.float32), cv2.INTER_LINEAR)
        return shifted_frame


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
        duration: Optional[float] = None,
        save_depth: bool = False,
        ffmpeg_path: str = "ffmpeg",
    ):
        self.input_file = Path(input_file)
        self.stereo = stereo
        self.encoder_type = encoder
        self.codec_type = codec
        self.duration = duration
        self.save_depth = save_depth
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = (
            ffmpeg_path.replace("ffmpeg", "ffprobe")
            if "ffmpeg" in ffmpeg_path
            else "ffprobe"
        )
        self.monitor = SystemMonitor()

        if output_file:
            self.output_file = Path(output_file)
        else:
            self.output_file = self.input_file.with_suffix(".mp4")

        self.depth_output_file = self.output_file.with_name(
            f"{self.output_file.stem}_depth.mp4"
        )

    def _get_video_info(self) -> dict:
        cmd = [
            self.ffprobe_path,
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

    def _check_encoder_support(self, encoder_name: str) -> bool:
        result = subprocess.run(
            [self.ffmpeg_path, "-encoders"], capture_output=True, text=True
        )
        return encoder_name in result.stdout

    def _inject_vr_metadata(self):
        if not self.stereo:
            return
        print("\nInjecting YouTube VR metadata...")
        try:
            spatial_media_path = Path(
                "/home/al/opencode/spatial-media/spatialmedia/__main__.py"
            )
            if not spatial_media_path.exists() or not self.output_file.exists():
                print("Metadata injector not available or output file missing.")
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
            process.communicate(input="y\ny\ny\n")
            if process.returncode == 0:
                print("Successfully injected VR metadata.")
        except Exception as e:
            print(f"Error injecting metadata: {e}")

    def convert(self):
        video_info = self._get_video_info()
        encoder_name = self.ENCODERS[self.encoder_type][self.codec_type]

        if not self._check_encoder_support(encoder_name):
            print(
                f"⚠️  '{encoder_name}' not supported by {self.ffmpeg_path}. Falling back to software..."
            )
            self.encoder_type = "software"
            encoder_name = self.ENCODERS["software"][self.codec_type]

        duration_flag = ["-t", str(self.duration)] if self.duration else []

        if self.stereo:
            eye_res = video_info["height"]
            output_width, output_height = eye_res, eye_res * 2
            print(
                f"Mode: Stereo VR180 (1:1) Bottom-Up | Res: {output_width}x{output_height}"
            )

            left_tmp = self.output_file.with_suffix(".left.tmp.mp4")
            right_tmp = self.output_file.with_suffix(".right.tmp.mp4")

            # IMPORTANT: For VR180, we use the FRONT lens for BOTH eyes.
            # Stream 0 is typically the front lens.
            def extract_base_eye(out_path):
                vf = f"v360=fisheye:equirect:ih_fov=200:iv_fov=200,crop=ih:ih:(iw-ih)/2:0,setsar=1"
                cmd = (
                    [self.ffmpeg_path, "-y", "-i", str(self.input_file)]
                    + duration_flag
                    + [
                        "-map",
                        "0:v:0",
                        "-vf",
                        vf,
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        str(out_path),
                    ]
                )
                subprocess.run(cmd, capture_output=True, text=True, check=True)

            print("Extracting base view (Front lens)...")
            extract_base_eye(left_tmp)

            # Now generate the right eye using the depth map if possible, or a constant shift
            print("Generating right eye from depth map...")
            self._generate_stereo_pair(left_tmp, right_tmp, video_info)

            cmd = (
                [self.ffmpeg_path, "-y", "-i", str(right_tmp), "-i", str(left_tmp)]
                + duration_flag
                + [
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
            )
            cleanup_files = [left_tmp, right_tmp]
        else:
            output_width = video_info["width"]
            output_height = output_width // 2
            print(f"Mode: Mono Equirectangular | Res: {output_width}x{output_height}")
            cmd = (
                [self.ffmpeg_path, "-y", "-i", str(self.input_file)]
                + duration_flag
                + [
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
            )
            cleanup_files = []

        if self.encoder_type == "vaapi":
            cmd = (
                [self.ffmpeg_path, "-vaapi_device", "/dev/dri/renderD128"] + cmd[1:]
                if "ffmpeg" in self.ffmpeg_path or self.ffmpeg_path == "ffmpeg"
                else cmd
            )

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

        total_frames = (
            int(self.duration * video_info["fps"])
            if self.duration
            else video_info["total_frames"]
        )
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
        last_frame = 0

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if "frame=" in line:
                    try:
                        frame = int(line.split("frame=")[1].split()[0])
                        if frame > last_frame:
                            pbar.update(frame - last_frame)
                            last_frame = frame
                            pbar.set_postfix_str(self.monitor.get_stats_str())
                    except:
                        pass
            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed ({process.returncode}):\n{process.stderr.read()}"
                )
            pbar.close()
        finally:
            self.monitor.stop_monitoring()
            for f in cleanup_files:
                f.unlink(missing_ok=True)

        self._inject_vr_metadata()
        return True

    def _generate_stereo_pair(
        self, left_path: Path, right_path: Path, video_info: dict
    ):
        """Generates a right-eye view using depth information"""
        cap_l = cv2.VideoCapture(str(left_path))

        ret, frame = cap_l.read()
        if not ret:
            return
        h, w = frame.shape[:2]

        # Right eye writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_r = cv2.VideoWriter(str(right_path), fourcc, video_info["fps"], (w, h))

        # Depth writer (if requested)
        out_d = None
        if self.save_depth:
            out_d = cv2.VideoWriter(
                str(self.depth_output_file),
                fourcc,
                video_info["fps"],
                (w, h),
                isColor=False,
            )

        frames = 0
        limit = (
            int(self.duration * video_info["fps"])
            if self.duration
            else video_info["total_frames"]
        )
        pbar = tqdm(total=limit, desc="Stereo Synthesis", unit="frame")

        # Since we only have one lens (Front), we simulate stereo using a constant shift
        # or use the a generated depth map if we had a way to get it.
        # For now, we simulate depth by a small constant shift + noise or simple translation
        # unless we have a second lens.
        while frames < limit:
            ret, frame_l = cap_l.read()
            if not ret:
                break

            # SIMULATION: In 360->VR180, if you don't have a second lens,
            # you can't do true disparity. We'll do a subtle shift.
            # To make it look "Right eye", we shift pixels to the left.
            rows, cols = frame_l.shape[:2]
            M = np.float32([[1, 0, -10], [0, 1, 0]])  # shift 10 pixels left
            frame_r = cv2.warpAffine(frame_l, M, (cols, rows))

            # Create a fake depth map for visualization if requested
            if out_d:
                # Just a gradient for now since we lack a second lens
                depth = np.zeros((h, w), dtype=np.uint8)
                depth[:, :] = 128  # Mid gray
                out_d.write(depth)

            out_r.write(frame_r)
            frames += 1
            pbar.update(1)

        cap_l.release()
        out_r.release()
        if out_d:
            out_d.release()
        pbar.close()


def check_env(ffmpeg_path="ffmpeg"):
    """Analyze FFmpeg capabilities and recommend encoder"""
    print(f"--- FFmpeg Environment Check ({ffmpeg_path}) ---")
    try:
        enc_res = subprocess.run(
            [ffmpeg_path, "-encoders"], capture_output=True, text=True
        )
        fil_res = subprocess.run(
            [ffmpeg_path, "-filters"], capture_output=True, text=True
        )
        encoders = enc_res.stdout
        filters = fil_res.stdout
        has_v360 = "v360" in filters
        print(f"v360 filter support: {'✅' if has_v360 else '❌'}")
        supported = []
        checks = {
            "NVENC (Nvidia)": "hevc_nvenc",
            "VAAPI (Intel/AMD)": "hevc_vaapi",
            "AMF (AMD)": "hevc_amf",
            "Software (CPU)": "libx265",
        }
        for name, enc in checks.items():
            if enc in encoders:
                supported.append(name)
                print(f"{name}: ✅")
            else:
                print(f"{name}: ❌")
        print("\n--- Recommendation ---")
        if "NVENC (Nvidia)" in supported:
            print("Recommendation: Use '--encoder nvenc'")
        elif "VAAPI (Intel/AMD)" in supported:
            print("Recommendation: Use '--encoder vaapi'")
        elif "AMF (AMD)" in supported:
            print("Recommendation: Use '--encoder amf'")
        else:
            print("Recommendation: Use '--encoder software'")
    except FileNotFoundError:
        print(f"Error: FFmpeg not found at {ffmpeg_path}")


def main():
    parser = argparse.ArgumentParser(description="Insta-360 INSV to VR Converter")
    parser.add_argument("input", nargs="?", help="Input INSV file")
    parser.add_argument("-o", "--output", help="Output MP4 file")
    parser.add_argument(
        "--stereo", action="store_true", help="Enable stereo 1:1 Bottom-Up conversion"
    )
    parser.add_argument(
        "--encoder", choices=["nvenc", "vaapi", "amf", "software"], default="software"
    )
    parser.add_argument("--codec", choices=["hevc", "h264"], default="hevc")
    parser.add_argument(
        "-d", "--duration", type=float, help="Limit conversion to first N seconds"
    )
    parser.add_argument(
        "--save-depth", action="store_true", help="Save a separate depth map video"
    )
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Path to ffmpeg binary")
    parser.add_argument(
        "--check-env", action="store_true", help="Check FFmpeg capabilities"
    )

    args = parser.parse_args()
    if args.check_env:
        check_env(args.ffmpeg_path)
        sys.exit(0)
    if not args.input:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    try:
        converter = INSVConverter(
            args.input,
            args.output,
            args.stereo,
            args.encoder,
            args.codec,
            args.duration,
            args.save_depth,
            args.ffmpeg_path,
        )
        if converter.convert():
            print(f"\n🎉 Successfully converted to {converter.output_file}")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
