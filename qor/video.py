"""
QOR Video — Frame Extraction from Files & URLs
=================================================
Extracts frames from video files and URLs for processing through
the vision pipeline (SigLIP or custom encoder).

Supports:
  - Local video files via OpenCV (mp4, avi, mkv, webm, mov)
  - Video URLs via yt-dlp (YouTube, Twitter, etc.)

Each extracted frame is a regular image that can be fed through
the vision encoder pipeline.

Usage as a tool:
    result = read_video("video.mp4")
    result = read_video("https://youtube.com/watch?v=...")

All dependencies are optional — graceful fallback with install hints.
"""

import os
import logging
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default output directory for extracted frames
_DEFAULT_SCREENSHOTS_DIR = os.path.join("qor-data", "screenshots")


class VideoFrameExtractor:
    """
    Extracts frames from video files and URLs.

    Frames are saved as PNG images and can be fed through
    the vision pipeline (SigLIP sees each frame).
    """

    def __init__(self, max_frames: int = 8, frame_interval_sec: float = 2.0,
                 screenshots_dir: str = _DEFAULT_SCREENSHOTS_DIR):
        self.max_frames = max_frames
        self.frame_interval_sec = frame_interval_sec
        self.screenshots_dir = screenshots_dir
        os.makedirs(screenshots_dir, exist_ok=True)

    def extract_from_file(self, path: str) -> List[str]:
        """
        Extract frames from a local video file using OpenCV.

        Args:
            path: Path to video file

        Returns:
            List of saved frame PNG paths
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        return self._extract_frames(path)

    def extract_from_url(self, url: str) -> List[str]:
        """
        Download video from URL via yt-dlp, then extract frames.

        Args:
            url: Video URL (YouTube, Twitter, etc.)

        Returns:
            List of saved frame PNG paths
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "Video URL download requires yt-dlp.\n"
                "Install: pip install yt-dlp"
            )

        # Download to temp file
        tmp_dir = tempfile.mkdtemp(prefix="qor_video_")
        output_path = os.path.join(tmp_dir, "video.mp4")

        ydl_opts = {
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(f"Failed to download video from {url}: {e}")

        # Find the downloaded file (yt-dlp may change extension)
        if not os.path.isfile(output_path):
            # Check for any video file in tmp_dir
            for f in os.listdir(tmp_dir):
                if f.endswith((".mp4", ".webm", ".mkv")):
                    output_path = os.path.join(tmp_dir, f)
                    break

        if not os.path.isfile(output_path):
            raise RuntimeError(f"Download completed but video file not found in {tmp_dir}")

        frames = self._extract_frames(output_path)

        # Clean up temp video file (keep extracted frames)
        try:
            os.remove(output_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

        return frames

    def _extract_frames(self, video_path: str) -> List[str]:
        """
        Core frame extraction: open with cv2, sample every N seconds, save as PNG.

        Args:
            video_path: Path to video file

        Returns:
            List of saved frame PNG paths
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video frame extraction requires OpenCV.\n"
                "Install: pip install opencv-python"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        # Calculate frame interval in frame numbers
        frame_interval = int(fps * self.frame_interval_sec)
        if frame_interval < 1:
            frame_interval = 1

        # Generate video name from path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Sanitize name
        video_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_name)

        saved_paths = []
        frame_idx = 0
        saved_count = 0

        while saved_count < self.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as PNG
            frame_path = os.path.join(
                self.screenshots_dir,
                f"video_{video_name}_frame_{saved_count}.png"
            )
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            saved_count += 1
            frame_idx += frame_interval

        cap.release()

        logger.info(f"Extracted {len(saved_paths)} frames from {video_path} "
                     f"(duration: {duration_sec:.1f}s, fps: {fps:.1f})")

        return saved_paths


def read_video(query: str) -> str:
    """
    Extract frames from a video file or URL.

    Tool handler for the QOR tool system.

    Args:
        query: Video file path or URL

    Returns:
        Text summary with frame count, paths, and video info
    """
    query = query.strip()

    # Strip common prefixes
    for prefix in ["watch ", "analyze ", "extract frames from ", "video "]:
        if query.lower().startswith(prefix):
            query = query[len(prefix):].strip()

    # Detect URL vs file path
    is_url = query.startswith("http://") or query.startswith("https://")

    extractor = VideoFrameExtractor()

    try:
        if is_url:
            frames = extractor.extract_from_url(query)
            source = query
        else:
            frames = extractor.extract_from_file(query)
            source = os.path.basename(query)
    except ImportError as e:
        return str(e)
    except FileNotFoundError:
        return f"Video file not found: {query}"
    except RuntimeError as e:
        return f"Video processing error: {e}"
    except Exception as e:
        return f"Failed to process video: {e}"

    if not frames:
        return f"No frames could be extracted from: {source}"

    # Build summary
    lines = [
        f"Extracted {len(frames)} frames from: {source}",
        f"Frames saved to: {extractor.screenshots_dir}/",
        "",
        "Frame paths:",
    ]
    for i, path in enumerate(frames):
        lines.append(f"  {i+1}. {path}")

    lines.append("")
    lines.append("These frames can be processed through the vision pipeline "
                  "for image understanding.")

    return "\n".join(lines)
