"""
Model training and evaluation module
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import config
from .utils import plot_training_history


class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, model):
        """
        Initialize trainer with a model
        
        Args:
            model: Keras model to train
        """
        self.model = model
        self.history = None
        self.train_metrics = {}
        self.test_metrics = {}
    
    def train(self, X_train, y_train, X_val, y_val, datagen=None):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            datagen: Data augmentation generator (optional)
        
        Returns:
            history: Training history
        """
        print("\n" + "="*60)
        print("Starting Model Training")
        print("="*60)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"{config.PATHS['models_dir']}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        if datagen:
            self.history = self.model.fit(
                datagen.flow(
                    X_train, y_train,
                    batch_size=config.TRAINING_CONFIG['batch_size']
                ),
                epochs=config.TRAINING_CONFIG['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=config.TRAINING_CONFIG['batch_size'],
                epochs=config.TRAINING_CONFIG['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
        
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating Model on Test Set")
        print("="*60)
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        self.test_metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        print(f"\nTest Loss: {self.test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {self.test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {self.test_metrics['precision']:.4f}")
        print(f"Test Recall: {self.test_metrics['recall']:.4f}")
        
        return self.test_metrics
    
    def get_predictions(self, X):
        """
        Get model predictions
        
        Args:
            X: Input images
        
        Returns:
            predictions: Model predictions
        """
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
