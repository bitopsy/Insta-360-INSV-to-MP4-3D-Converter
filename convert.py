#!/usr/bin/env python3
"""
INSV to Equirectangular Converter with HEVC CUDA Support
Converts spherical INSV format to equirectangular 2:1 format with optional depth map generation
"""

import os
import sys
import time
import threading
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple
import psutil
import GPUtil
import cv2
import numpy as np
from datetime import datetime, timedelta

class SystemMonitor:
    """Monitor CPU, memory, and VRAM usage"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'vram_percent': [],
            'vram_used_mb': []
        }
        
    def start_monitoring(self):
        """Start system monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU/VRAM
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    vram_percent = gpu.memoryUtil * 100
                    vram_used_mb = gpu.memoryUsed
                else:
                    vram_percent = 0
                    vram_used_mb = 0
            except:
                vram_percent = 0
                vram_used_mb = 0
                
            self.stats['cpu_percent'].append(cpu_percent)
            self.stats['memory_percent'].append(memory.percent)
            self.stats['vram_percent'].append(vram_percent)
            self.stats['vram_used_mb'].append(vram_used_mb)
            
            time.sleep(1)
            
    def get_current_stats(self) -> dict:
        """Get current system statistics"""
        if not self.stats['cpu_percent']:
            return {'cpu': 0, 'memory': 0, 'vram': 0, 'vram_mb': 0}
            
        return {
            'cpu': self.stats['cpu_percent'][-1],
            'memory': self.stats['memory_percent'][-1],
            'vram': self.stats['vram_percent'][-1],
            'vram_mb': self.stats['vram_used_mb'][-1]
        }
        
    def get_average_stats(self) -> dict:
        """Get average system statistics"""
        if not self.stats['cpu_percent']:
            return {'cpu': 0, 'memory': 0, 'vram': 0, 'vram_mb': 0}
            
        return {
            'cpu': np.mean(self.stats['cpu_percent']),
            'memory': np.mean(self.stats['memory_percent']),
            'vram': np.mean(self.stats['vram_percent']),
            'vram_mb': np.mean(self.stats['vram_used_mb'])
        }

class DepthMapGenerator:
    """Generate estimated depth maps for stereo 3D conversion"""
    
    @staticmethod
    def generate_depth_from_motion(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Generate depth map using optical flow between consecutive frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, 
            np.random.rand(1000, 1, 2).astype(np.float32) * [frame1.shape[1], frame1.shape[0]],
            None
        )[0]
        
        # Create depth map based on motion magnitude
        depth_map = np.zeros_like(gray1, dtype=np.float32)
        
        # Simple depth estimation (objects with more motion are closer)
        if flow is not None:
            for point in flow:
                if point is not None:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                        motion_mag = np.linalg.norm(point)
                        depth_map[y, x] = min(255, motion_mag * 10)
        
        # Smooth and interpolate
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        depth_map = cv2.resize(depth_map, (frame1.shape[1], frame1.shape[0]))
        
        return depth_map.astype(np.uint8)
    
    @staticmethod
    def generate_depth_from_gradient(frame: np.ndarray) -> np.ndarray:
        """Generate depth map using image gradients (objects in focus are closer)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur for smoother depth transitions
        depth_map = cv2.GaussianBlur(depth_map, (21, 21), 0)
        
        return depth_map.astype(np.uint8)

class INSVConverter:
    """Main converter class for INSV to equirectangular format"""
    
    def __init__(self, input_file: str, output_file: str, generate_depth: bool = False):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.generate_depth = generate_depth
        self.monitor = SystemMonitor()
        self.depth_generator = DepthMapGenerator()
        
        # Check if input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
        # Check for CUDA support
        self.cuda_available = self._check_cuda_support()
        
    def _check_cuda_support(self) -> bool:
        """Check if NVIDIA GPU and CUDA are available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def _get_video_info(self) -> dict:
        """Get video information using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            str(self.input_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # INSV files typically have dual fisheye streams
            video_streams = [s for s in info['streams'] if s['codec_type'] == 'video']
            if not video_streams:
                raise ValueError("No video streams found")
                
            # Use first video stream for basic info
            video_stream = video_streams[0]
            
            # Calculate frame rate properly
            fps_str = video_stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            
            duration = float(info['format'].get('duration', 0))
            total_frames = int(duration * fps) if duration > 0 else 0
            
            return {
                'duration': duration,
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'total_frames': total_frames,
                'stream_count': len(video_streams),
                'pixel_format': video_stream.get('pix_fmt', 'yuv420p')
            }
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get video info: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse video info: {e}")
            
    def _calculate_output_resolution(self, input_width: int, input_height: int) -> Tuple[int, int]:
        """Calculate appropriate output resolution maintaining 2:1 aspect ratio"""
        # For INSV files, the input is typically dual fisheye
        # The output equirectangular should be 2:1 aspect ratio
        
        # Base the output resolution on input resolution but ensure 2:1 ratio
        # Common resolutions: 1920x960, 2880x1440, 3840x1920, 5760x2880, 7680x3840
        
        # Calculate based on input dimensions - use the larger dimension as reference
        reference_dim = max(input_width, input_height)
        
        if reference_dim <= 1920:
            return (1920, 960)    # 1080p base
        elif reference_dim <= 2880:
            return (2880, 1440)   # 1440p base  
        elif reference_dim <= 3840:
            return (3840, 1920)   # 4K base
        elif reference_dim <= 5760:
            return (5760, 2880)   # 6K base
        else:
            return (7680, 3840)   # 8K base
            
    def _build_ffmpeg_command(self, video_info: dict, temp_depth_file: Optional[str] = None) -> list:
        """Build FFmpeg command for conversion"""
        
        # Calculate output resolution
        output_width, output_height = self._calculate_output_resolution(
            video_info['width'], video_info['height']
        )
        
        print(f"Output resolution: {output_width}x{output_height} (2:1 aspect ratio)")
        
        # Base command with input
        cmd = ['ffmpeg', '-y', '-i', str(self.input_file)]
        
        # For INSV files, we need to handle the dual fisheye properly
        # INSV typically stores dual fisheye in side-by-side or separate streams
        
        if video_info['stream_count'] >= 2:
            # Multiple video streams (separate fisheye inputs)
            video_filter = (
                f"[0:v:0]v360=fisheye:equirect:ih_fov=200:iv_fov=200,"
                f"scale={output_width//2}:{output_height},setsar=1[front];"
                f"[0:v:1]v360=fisheye:equirect:ih_fov=200:iv_fov=200,"
                f"scale={output_width//2}:{output_height},setsar=1[back];"
                f"[front][back]hstack=inputs=2[equirect];"
                f"[equirect]scale={output_width}:{output_height},setsar=1"
            )
        else:
            # Single stream with side-by-side fisheye
            video_filter = (
                f"[0:v]crop=w=iw/2:h=ih:x=0:y=0,"
                f"v360=fisheye:equirect:ih_fov=200:iv_fov=200,"
                f"scale={output_width//2}:{output_height},setsar=1[left];"
                f"[0:v]crop=w=iw/2:h=ih:x=iw/2:y=0,"
                f"v360=fisheye:equirect:ih_fov=200:iv_fov=200,"
                f"scale={output_width//2}:{output_height},setsar=1[right];"
                f"[left][right]hstack=inputs=2[equirect];"
                f"[equirect]scale={output_width}:{output_height},setsar=1"
            )
        
        if self.generate_depth and temp_depth_file:
            # Add depth map input and create top-bottom 3D format
            cmd.extend(['-i', temp_depth_file])
            video_filter += f"[main];[1:v]scale={output_width}:{output_height},setsar=1[depth];[main][depth]vstack=inputs=2"
            final_height = output_height * 2  # Top-bottom format doubles height
        else:
            final_height = output_height
            
        cmd.extend(['-filter_complex', video_filter])
        
        # Force pixel format to avoid color issues
        cmd.extend(['-pix_fmt', 'yuv420p'])
        
        # HEVC encoding with CUDA if available
        if self.cuda_available:
            cmd.extend([
                '-c:v', 'hevc_nvenc',
                '-preset', 'medium',
                '-cq', '23',
                '-b:v', f'{max(10, output_width * output_height // 50000)}M',  # Dynamic bitrate
                '-maxrate', f'{max(15, output_width * output_height // 35000)}M',
                '-bufsize', f'{max(30, output_width * output_height // 15000)}M',
                '-rc', 'vbr',
                '-rc-lookahead', '20'
            ])
            print("Using NVIDIA HEVC hardware encoding")
        else:
            cmd.extend([
                '-c:v', 'libx265',
                '-preset', 'medium',
                '-crf', '23',
                '-x265-params', 'log-level=error'
            ])
            print("Using software HEVC encoding (CUDA not available)")
            
        # Audio codec - copy if available, otherwise skip
        cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        
        # Ensure proper format and avoid pink coloring
        cmd.extend([
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-fflags', '+genpts'
        ])
        
        # Progress reporting
        cmd.extend(['-progress', 'pipe:1', '-nostats'])
        
        # Output file
        cmd.append(str(self.output_file))
        
        return cmd
        
    def _create_depth_video(self, video_info: dict) -> str:
        """Create depth map video from input tracks"""
        temp_depth_file = str(self.output_file.with_suffix('.depth.mp4'))
        
        print("Generating depth map video from stereo tracks...")
        
        # Calculate output resolution for depth map
        output_width, output_height = self._calculate_output_resolution(
            video_info['width'], video_info['height']
        )
        
        if video_info['stream_count'] >= 2:
            # Multiple streams - extract both for stereo depth
            temp_track1 = str(self.output_file.with_suffix('.track1.mp4'))
            temp_track2 = str(self.output_file.with_suffix('.track2.mp4'))
            
            # Extract and convert track 1 to equirectangular
            cmd1 = [
                'ffmpeg', '-y', '-i', str(self.input_file), '-map', '0:v:0',
                '-vf', f'v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width//2}:{output_height}',
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', temp_track1
            ]
            subprocess.run(cmd1, capture_output=True)
            
            # Extract and convert track 2 to equirectangular
            cmd2 = [
                'ffmpeg', '-y', '-i', str(self.input_file), '-map', '0:v:1',
                '-vf', f'v360=fisheye:equirect:ih_fov=200:iv_fov=200,scale={output_width//2}:{output_height}',
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', temp_track2
            ]
            subprocess.run(cmd2, capture_output=True)
            
            # Process frames for depth generation
            cap1 = cv2.VideoCapture(temp_track1)
            cap2 = cv2.VideoCapture(temp_track2)
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                temp_depth_file,
                fourcc,
                video_info['fps'],
                (output_width, output_height),
                isColor=False
            )
            
            frame_count = 0
            
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not (ret1 and ret2):
                    break
                    
                # Generate depth map from stereo pair
                depth_map = self._generate_stereo_depth(frame1, frame2)
                # Resize to match output resolution
                depth_map = cv2.resize(depth_map, (output_width, output_height))
                out.write(depth_map)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / video_info['total_frames']) * 100
                    print(f"Depth generation progress: {progress:.1f}%")
                    
            cap1.release()
            cap2.release()
            out.release()
            
            # Clean up temporary files
            for temp_file in [temp_track1, temp_track2]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            # Single stream - use gradient-based depth estimation
            print("Single stream detected, using gradient-based depth estimation...")
            
            # Extract frames and generate depth maps
            cap = cv2.VideoCapture(str(self.input_file))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                temp_depth_file,
                fourcc,
                video_info['fps'],
                (output_width, output_height),
                isColor=False
            )
            
            frame_count = 0
            prev_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if prev_frame is not None:
                    # Use motion-based depth
                    depth_map = self.depth_generator.generate_depth_from_motion(prev_frame, frame)
                else:
                    # Use gradient-based depth for first frame
                    depth_map = self.depth_generator.generate_depth_from_gradient(frame)
                    
                # Resize to match output resolution
                depth_map = cv2.resize(depth_map, (output_width, output_height))
                out.write(depth_map)
                
                prev_frame = frame
                frame_count += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / video_info['total_frames']) * 100
                    print(f"Depth generation progress: {progress:.1f}%")
                    
            cap.release()
            out.release()
        
        return temp_depth_file
        
    def _generate_stereo_depth(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """Generate depth map from stereo pair using disparity calculation"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Create stereo matcher with better parameters
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=21)
        stereo.setPreFilterCap(31)
        stereo.setBlockSize(21)
        stereo.setMinDisparity(0)
        stereo.setNumDisparities(96)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)
        
        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right)
        
        # Normalize disparity to depth map (0-255)
        depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
        depth_map = cv2.medianBlur(depth_map, 5)
        
        return depth_map
        
    def _parse_ffmpeg_progress(self, line: str) -> Optional[dict]:
        """Parse FFmpeg progress output"""
        progress_data = {}
        
        # Parse key=value pairs from FFmpeg progress
        if '=' in line:
            key, value = line.strip().split('=', 1)
            if key == 'frame':
                try:
                    progress_data['frame'] = int(value)
                except ValueError:
                    pass
            elif key == 'out_time_ms':
                try:
                    # Convert microseconds to seconds
                    progress_data['time_seconds'] = int(value) / 1000000
                except ValueError:
                    pass
            elif key == 'progress':
                progress_data['status'] = value
                
        return progress_data if progress_data else None
        
    def convert(self):
        """Main conversion function"""
        print(f"Converting {self.input_file.name} to equirectangular format...")
        print(f"CUDA available: {self.cuda_available}")
        
        # Get video information
        try:
            video_info = self._get_video_info()
            output_width, output_height = self._calculate_output_resolution(
                video_info['width'], video_info['height']
            )
            
            print(f"Input: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['fps']:.2f}fps, {video_info['duration']:.1f}s")
            print(f"Output: {output_width}x{output_height} (2:1 ratio), "
                  f"{video_info['stream_count']} input streams")
        except Exception as e:
            print(f"Error getting video info: {e}")
            return False
            
        # Start system monitoring
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Generate depth map video if requested
            temp_depth_file = None
            if self.generate_depth:
                temp_depth_file = self._create_depth_video(video_info)
                
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(video_info, temp_depth_file)
            print(f"FFmpeg command: {' '.join(cmd[:10])}...")  # Truncated for readability
            
            # Execute conversion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor progress
            last_progress_time = time.time()
            frames_processed = 0
            current_time_seconds = 0
            
            # Read progress from stdout
            for line in iter(process.stdout.readline, ''):
                progress_info = self._parse_ffmpeg_progress(line)
                if progress_info:
                    if 'frame' in progress_info:
                        frames_processed = progress_info['frame']
                    if 'time_seconds' in progress_info:
                        current_time_seconds = progress_info['time_seconds']
                        
                    # Update progress every 2 seconds
                    current_time = time.time()
                    if current_time - last_progress_time >= 2:
                        self._print_progress(frames_processed, video_info['total_frames'], 
                                           current_time_seconds, video_info['duration'],
                                           start_time, current_time)
                        last_progress_time = current_time
                        
                if progress_info and progress_info.get('status') == 'end':
                    break
                    
            # Wait for process to complete
            return_code = process.wait()
            
            # Clean up temporary depth file
            if temp_depth_file and os.path.exists(temp_depth_file):
                os.remove(temp_depth_file)
                
            if return_code == 0:
                print("\n✅ Conversion completed successfully!")
                self._print_final_stats(start_time)
                return True
            else:
                stderr = process.stderr.read()
                print(f"\n❌ Conversion failed with return code {return_code}")
                print(f"Error: {stderr}")
                return False
                
        except KeyboardInterrupt:
            print("\n🛑 Conversion interrupted by user")
            if 'process' in locals():
                process.terminate()
            return False
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            return False
        finally:
            self.monitor.stop_monitoring()
            
    def _print_progress(self, frames_processed: int, total_frames: int,
                      current_time_seconds: float, total_duration: float,
                      start_time: float, current_time: float):
        """Print conversion progress with system stats"""
        
        # Calculate progress percentage
        if total_duration > 0 and current_time_seconds > 0:
            progress_percent = min(100, (current_time_seconds / total_duration) * 100)
        elif total_frames > 0 and frames_processed > 0:
            progress_percent = min(100, (frames_processed / total_frames) * 100)
        else:
            progress_percent = 0
            
        elapsed_time = current_time - start_time
        
        # Estimate remaining time
        if progress_percent > 1:
            eta_seconds = (elapsed_time / progress_percent) * (100 - progress_percent)
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "Unknown"
            
        # Get current system stats
        stats = self.monitor.get_current_stats()
        
        print(f"\r🎬 Progress: {progress_percent:.1f}% "
              f"({current_time_seconds:.1f}s/{total_duration:.1f}s) | "
              f"⏱️  ETA: {eta} | "
              f"🔥 CPU: {stats['cpu']:.1f}% | "
              f"💾 RAM: {stats['memory']:.1f}% | "
              f"🎮 VRAM: {stats['vram']:.1f}% ({stats['vram_mb']:.0f}MB)", end='')
              
    def _print_final_stats(self, start_time: float):
        """Print final conversion statistics"""
        total_time = time.time() - start_time
        avg_stats = self.monitor.get_average_stats()
        
        print(f"\n📊 Conversion Statistics:")
        print(f"   ⏰ Total time: {timedelta(seconds=int(total_time))}")
        print(f"   🔥 Average CPU usage: {avg_stats['cpu']:.1f}%")
        print(f"   💾 Average RAM usage: {avg_stats['memory']:.1f}%")
        print(f"   🎮 Average VRAM usage: {avg_stats['vram']:.1f}% ({avg_stats['vram_mb']:.0f}MB)")
        
        # File size info
        if self.output_file.exists():
            output_size = self.output_file.stat().st_size / (1024**3)  # GB
            input_size = self.input_file.stat().st_size / (1024**3)   # GB
            compression_ratio = (1 - output_size/input_size) * 100 if input_size > 0 else 0
            
            print(f"   📁 Input size: {input_size:.2f} GB")
            print(f"   📁 Output size: {output_size:.2f} GB")
            print(f"   📉 Size change: {compression_ratio:+.1f}%")

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'ffmpeg': 'FFmpeg is required for video processing',
        'ffprobe': 'FFprobe is required for video analysis'
    }
    
    missing = []
    for dep, desc in dependencies.items():
        try:
            result = subprocess.run([dep, '-version'], capture_output=True, check=True)
            # Check for v360 filter support
            if dep == 'ffmpeg':
                if b'v360' not in result.stderr and b'v360' not in result.stdout:
                    # Test v360 filter availability
                    test_cmd = [dep, '-hide_banner', '-filters']
                    test_result = subprocess.run(test_cmd, capture_output=True, text=True)
                    if 'v360' not in test_result.stdout:
                        missing.append(f"FFmpeg v360 filter: Required for 360° video conversion")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(f"{dep}: {desc}")
            
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\nPlease install FFmpeg with v360 filter support and ensure it's in your PATH.")
        return False
        
    # Check Python packages
    try:
        import cv2, numpy, psutil, GPUtil
        print("✅ All Python dependencies found")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("Install with: pip install opencv-python numpy psutil GPUtil")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert INSV spherical video to equirectangular format with HEVC CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python insv_converter.py input.insv output.mp4
  python insv_converter.py input.insv output.mp4 --depth
  python insv_converter.py --check-deps
        """
    )
    parser.add_argument('input', nargs='?', help='Input INSV file path')
    parser.add_argument('output', nargs='?', help='Output video file path')
    parser.add_argument('--depth', action='store_true', 
                       help='Generate depth map for top-bottom 3D format')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are available!")
            
            # Also check for CUDA
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ NVIDIA CUDA support detected")
                else:
                    print("⚠️  CUDA not available - will use software encoding")
            except FileNotFoundError:
                print("⚠️  CUDA not available - will use software encoding")
        sys.exit(0)
        
    # Validate required arguments
    if not args.input or not args.output:
        parser.print_help()
        print("\n❌ Both input and output file paths are required")
        sys.exit(1)
        
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
        
    # Validate input file extension
    if not args.input.lower().endswith('.insv'):
        print(f"⚠️  Warning: Input file doesn't have .insv extension")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
        
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize converter
        converter = INSVConverter(args.input, args.output, args.depth)
        
        print(f"🚀 Starting conversion...")
        print(f"📥 Input: {args.input}")
        print(f"📤 Output: {args.output}")
        
        if args.depth:
            print("🎯 Depth map generation: Enabled (Top-Bottom 3D format)")
        else:
            print("🎯 Output format: Equirectangular 2:1")
            
        # Start conversion
        success = converter.convert()
        
        if success:
            print(f"🎉 Video successfully converted to {args.output}")
            
            # Verify output file
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"📁 Output file size: {size_mb:.1f} MB")
            
            sys.exit(0)
        else:
            print("💥 Conversion failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Conversion interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
