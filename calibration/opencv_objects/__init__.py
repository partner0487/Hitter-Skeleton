"""
This module initializes the OpenCV objects package.

Classes:
    ChessboardCalibrator: A class for calibrating stereo cameras.

__all__:
    List of public objects of that module, as interpreted by import *.
"""

from .chessboard_calibration import ChessboardCalibrator

__all__ = ["ChessboardCalibrator"]
