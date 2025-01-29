"""
Module for loading the trained neural network model.
"""
import os
import tensorflow as tf

def model_fun() -> tf.keras.Model:
    """
    Load the pre-trained CNN model for pneumonia detection.
    
    Returns:
        tf.keras.Model: Loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For other loading errors
    """
    model_path = 'data/external/models/conv_MLP_84.h5'
    
    # Verificar que el archivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    try:
        # Cargar el modelo con compile=False para evitar advertencias
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompilar el modelo con configuraciones básicas
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Verificar que el modelo tiene la capa esperada
        if not any(layer.name == "conv10_thisone" for layer in model.layers):
            raise ValueError("El modelo cargado no tiene la capa 'conv10_thisone' requerida para Grad-CAM")
        
        return model
        
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {str(e)}")