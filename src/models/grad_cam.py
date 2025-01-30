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
    """
    # Preprocesar imagen y convertirla a tensor
    img = preprocess(array)
    img_tensor = tf.convert_to_tensor(img)
    
    # Cargar modelo
    model = model_fun()
    
    # Crear un modelo que mapee la entrada a la capa convolucional y la salida
    last_conv_layer = model.get_layer("conv10_thisone")
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Calcular predicciones y gradientes usando GradientTape
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_output, predictions = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Obtener gradientes
    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise ValueError("No se pudieron calcular los gradientes.")
    
    # Calcular los pesos de importancia para cada filtro
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal por su peso de importancia y sumar
    conv_output = conv_output[0]  # Primera imagen del batch
    pooled_grads = tf.reshape(pooled_grads, (1, 1, -1))
    weighted_conv = tf.multiply(conv_output, pooled_grads)
    
    # Crear el mapa de calor
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()
    
    # Redimensionar a las dimensiones objetivo
    target_size = (512, 512)
    heatmap = cv2.resize(heatmap, target_size)
    
    # Normalizar a valores de 0-255 y convertir a uint8
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Preparar la imagen original
    img2 = cv2.resize(array, (512, 512))
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2 = img2.astype(np.uint8)
    
    # Superponer el mapa de calor
    alpha = 0.7
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img2, 1 - alpha, 0)
    
    return superimposed_img[:, :, ::-1]
