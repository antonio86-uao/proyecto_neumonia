"""
Module for integrating model components and providing interface functionality.
"""
from ..data.read_img import read_dicom_file
from ..data.preprocess_img import preprocess  
from ..models.load_model import model_fun
from ..models.grad_cam import grad_cam
import numpy as np


def predict(array: np.ndarray) -> tuple:
    """
    Integrate model components to generate predictions.
    
    Args:
        array (np.ndarray): Input image array
        
    Returns:
        tuple: (prediction_label, probability, heatmap)
    """
    batch_array_img = preprocess(array)
    model = model_fun()
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    
    label = {
        0: "bacteriana",
        1: "normal",
        2: "viral"
    }.get(prediction, "")
    
    heatmap = grad_cam(array)
    
    return label, proba, heatmap