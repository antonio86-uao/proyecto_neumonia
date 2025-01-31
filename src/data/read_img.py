"""
Módulo para leer y procesar imágenes médicas en formato DICOM y otros formatos.
"""

import pydicom as dicom
import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def read_dicom_file(path: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Lee y procesa un archivo DICOM.

    Args:
        path (str): Ruta del archivo DICOM.

    Returns:
        Tuple[np.ndarray, Image.Image]: Matriz de imagen procesada y 
        objeto PIL Image.
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
    Lee y procesa un archivo JPG/JPEG.

    Args:
        path (str): Ruta del archivo JPG.

    Returns:
        Tuple[np.ndarray, Image.Image]: Matriz de imagen procesada y 
        objeto PIL Image.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    return img2, img2show
