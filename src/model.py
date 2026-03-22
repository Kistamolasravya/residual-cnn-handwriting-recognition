"""
Residual CNN Model Architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


def residual_block(x, filters, kernel_size=3, stride=1, activation="relu"):
    """
    Create a residual block with two convolutional layers
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolution
        stride: Stride for convolution
        activation: Activation function
    
    Returns:
        Output tensor
    """
    shortcut = x
    
    # First convolutional layer
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding="same", name=f"conv_1"
    )(x)
    x = layers.BatchNormalization(name=f"bn_1")(x)
    x = layers.Activation(activation, name=f"relu_1")(x)
    
    # Second convolutional layer
    x = layers.Conv2D(
        filters, kernel_size, strides=1, padding="same", name=f"conv_2"
    )(x)
    x = layers.BatchNormalization(name=f"bn_2")(x)
    
    # Adjust shortcut if dimensions don't match
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", name=f"conv_shortcut"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"bn_shortcut")(shortcut)
    
    # Add shortcut and apply activation
    x = layers.Add(name=f"add")([x, shortcut])
    x = layers.Activation(activation, name=f"relu_out")(x)
    
    return x


def build_residual_cnn(input_shape, num_classes, num_blocks=[2, 2, 2, 2]):
    """
    Build a Residual CNN model for digit recognition
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        num_blocks: List of number of residual blocks per stage
    
    Returns:
        keras.Model: Compiled model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(
        config.MODEL_CONFIG['initial_filters'], 
        3, 
        strides=1, 
        padding="same",
        name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(2, name="maxpool1")(x)
    
    # Residual blocks
    filters = config.MODEL_CONFIG['initial_filters']
    for stage, num in enumerate(num_blocks):
        for block in range(num):
            stride = 2 if block == 0 and stage > 0 else 1
            filters = config.MODEL_CONFIG['initial_filters'] * (2 ** stage)
            block_name = f"block{stage+1}_{block+1}"
            
            x = residual_block(
                x, 
                filters=filters, 
                stride=stride,
                activation="relu"
            )
            # Rename layers for easier access
            x._name = f"{block_name}_output"
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D(name="global_avgpool")(x)
    
    # Fully connected layers
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.5, name="dropout2")(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResidualCNN")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.TRAINING_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model
