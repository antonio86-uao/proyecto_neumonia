"""
Module for loading the trained neural network model.
"""
import tensorflow as tf
from tensorflow.keras import backend as K

def model_fun() -> tf.keras.Model:
    """
    Load the pre-trained CNN model for pneumonia detection.
    
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model('data/external/models/conv_MLP_84.h5')
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")