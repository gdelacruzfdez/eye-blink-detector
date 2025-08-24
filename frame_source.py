"""Abstraction for providing video frames.

This module defines the :class:`FrameSource` protocol which describes the
interface required by classes that supply video frames to the application.
Implementations may obtain frames from a webcam, a video file or any other
source.  The interface exposes a minimal set of methods needed by the rest of
the application to interact with a frame provider.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class FrameSource(Protocol):
    """Protocol for objects that supply video frames.

    Classes implementing this protocol should provide frames as numpy arrays in
    BGR format (matching the output of ``cv2.VideoCapture``).  They also expose
    basic information about the stream such as its dimensions and frame rate and
    must release any held resources when ``release`` is called.
    """

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the next frame from the source or ``None`` if unavailable."""

    def get_frame_width(self) -> int:
        """Return the width of frames produced by the source."""

    def get_frame_height(self) -> int:
        """Return the height of frames produced by the source."""

    def get_fps(self) -> float:
        """Return the frame rate of the source in frames per second."""

    def release(self) -> None:
        """Release any resources held by the frame source."""

