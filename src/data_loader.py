"""
Data loading and preprocessing module for MNIST dataset
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def load_mnist_data():
    """
    Load MNIST dataset from Keras
    
    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Original dataset shapes:")
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)


def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Preprocess MNIST data: reshape, normalize, and convert labels to one-hot
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
    
    Returns:
        tuple: Preprocessed (X_train, y_train), (X_test, y_test)
    """
    # Reshape data to include channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    
    # Normalize pixel values
    if config.DATA_CONFIG['normalize']:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, config.MODEL_CONFIG['num_classes'])
    y_test = keras.utils.to_categorical(y_test, config.MODEL_CONFIG['num_classes'])
    
    print(f"Preprocessed dataset shapes:")
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    return (X_train, y_train), (X_test, y_test)


def get_data_augmentation():
    """
    Create data augmentation pipeline for training
    
    Returns:
        ImageDataGenerator: Configured data augmentation generator
    """
    if config.DATA_CONFIG['augmentation']:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        print("Data augmentation enabled")
    else:
        datagen = ImageDataGenerator()
        print("Data augmentation disabled")
    
    return datagen


def create_train_val_split(X_train, y_train, validation_split=0.1, random_state=42):
    """
    Split training data into training and validation sets
    
    Args:
        X_train: Training images
        y_train: Training labels
        validation_split: Fraction for validation
        random_state: Random seed
    
    Returns:
        tuple: (X_train, y_train), (X_val, y_val)
    """
    num_samples = len(X_train)
    indices = np.arange(num_samples)
    np.random.RandomState(random_state).shuffle(indices)
    
    split_idx = int(num_samples * (1 - validation_split))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    print(f"Train/Val split: {X_train_split.shape[0]}/{X_val.shape[0]}")
    
    return (X_train_split, y_train_split), (X_val, y_val)
