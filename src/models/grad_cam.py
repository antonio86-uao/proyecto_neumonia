"""
Module for generating Grad-CAM visualizations of model predictions.
"""
import numpy as np
import cv2
import tensorflow as tf
from .load_model import model_fun
from ..data.preprocess_img import preprocess

def grad_cam(array: np.ndarray) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model predictions.
    
    Args:
        array (np.ndarray): Input image array
        
    Returns:
        np.ndarray: Image with superimposed heatmap
    """
    img = preprocess(array)
    model = model_fun()
    
    # Crear un modelo que mapee la entrada a la capa convolucional y la salida
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("conv10_thisone").output, model.output]
    )
    
    # Calcular predicciones y gradientes usando GradientTape
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]
    
    # Obtener gradientes
    grads = tape.gradient(output, conv_output)
    
    # Calcular los pesos de importancia para cada filtro
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal por su peso de importancia
    conv_output = conv_output[0]
    weighted_conv = tf.multiply(pooled_grads, conv_output)
    
    # Crear el mapa de calor
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Redimensionar y aplicar el mapa de color
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer el mapa de calor en la imagen original
    img2 = cv2.resize(array, (512, 512))
    transparency = heatmap * 0.8
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    
    return superimposed_img[:, :, ::-1]