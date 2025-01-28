"""
Module for reading and processing medical images in DICOM and other formats.
"""
import pydicom as dicom
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Read and process a DICOM file.
    Args:
    path (str): Path to DICOM file
    Returns:
    Tuple[np.ndarray, Image.Image]: Processed image array and PIL Image
    """
    img = dicom.read_file(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show

def read_jpg_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Read and process a JPG/JPEG file.
    Args:
    path (str): Path to JPG file
    Returns:
    Tuple[np.ndarray, Image.Image]: Processed image array and PIL Image
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show