"""
Module for preprocessing medical images before model prediction.
"""
import cv2
import numpy as np
from numpy import ndarray

def preprocess(array: np.ndarray) -> np.ndarray:
    """
    Preprocess image array for model input.
    
    Args:
        array (np.ndarray): Input image array
    Returns:
        np.ndarray: Preprocessed image batch
        
    Processing steps:
        1. Resize to 512x512
        2. Convert to grayscale
        3. Apply CLAHE histogram equalization
        4. Normalize to [0,1]
        5. Convert to batch format
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array