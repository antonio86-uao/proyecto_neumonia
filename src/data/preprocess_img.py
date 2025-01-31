"""
Módulo para preprocesar imágenes médicas antes de la predicción del modelo.
"""

import cv2
import numpy as np


def preprocess(array: np.ndarray) -> np.ndarray:
    """
    Preprocesa una matriz de imagen para la entrada del modelo.

    Args:
        array (np.ndarray): Matriz de imagen de entrada.

    Returns:
        np.ndarray: Lote de imagen preprocesada.

    Pasos de procesamiento:
        1. Redimensionar a 512x512.
        2. Convertir a escala de grises.
        3. Aplicar ecualización de histograma CLAHE.
        4. Normalizar a [0,1].
        5. Convertir a formato de lote.
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
