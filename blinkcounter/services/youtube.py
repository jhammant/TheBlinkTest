"""YouTube video download service using yt-dlp."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

logger = logging.getLogger(__name__)

# Persistent cache directory for downloaded videos
_CACHE_DIR = Path.home() / ".cache" / "blinkcounter" / "videos"


def _get_cache_path(url: str) -> Path:
    """Get cache file path for a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{url_hash}.mp4"


def download_video(
    url: str,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Download a video from YouTube using yt-dlp.

    Downloads are cached in ~/.cache/blinkcounter/videos/ so repeated
    analysis of the same video doesn't re-download.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the video. Uses a temp directory if None.
        progress_callback: Optional callback receiving (progress 0.0-1.0, status).

    Returns:
        Path to the downloaded video file.

    Raises:
        yt_dlp.utils.DownloadError: If the download fails.
    """
    # Check cache first
    cache_path = _get_cache_path(url)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info("Cache hit: %s", cache_path)
        if progress_callback:
            progress_callback(1.0, "Cached")
        return str(cache_path)

    if output_dir is None:
        output_dir = str(_CACHE_DIR)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(_CACHE_DIR, exist_ok=True)

    def _progress_hook(d: dict) -> None:
        if progress_callback is None:
            return

        status = d.get("status", "")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                progress = min(downloaded / total, 1.0)
                progress_callback(progress, f"Downloading: {progress:.0%}")
            else:
                progress_callback(0.0, "Downloading...")
        elif status == "finished":
            progress_callback(1.0, "Download complete, processing...")

    ydl_opts = {
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
            "/best[height<=720][ext=mp4]"
            "/best"
        ),
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    if progress_callback:
        progress_callback(0.0, "Starting download...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)

    # Copy to cache for future reuse
    if filepath != str(cache_path) and os.path.exists(filepath):
        import shutil
        try:
            shutil.copy2(filepath, cache_path)
            logger.info("Cached video: %s", cache_path)
        except OSError:
            pass

    logger.info("Downloaded video to %s", filepath)

    if progress_callback:
        progress_callback(1.0, "Download complete")

    return filepath
