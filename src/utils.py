"""
Utility functions for visualization and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import config


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (probabilities or class indices)
        class_names: List of class names
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=class_names or range(10),
        yticklabels=class_names or range(10),
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_training_history(history, figsize=(14, 4)):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history from model.fit()
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_samples(images, labels, predictions=None, num_samples=16, figsize=(12, 8)):
    """
    Plot sample images with labels
    
    Args:
        images: Array of images
        labels: Array of labels
        predictions: Predicted labels (optional)
        num_samples: Number of samples to plot
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    """
    num_show = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_show)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_show):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        
        # Convert one-hot to class index if needed
        true_label = labels[i]
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            true_label = np.argmax(labels[i])
        
        title = f'True: {true_label}'
        if predictions is not None:
            pred_label = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}, Pred: {pred_label}\n({confidence:.2%})'
            axes[i].set_title(title, color=color)
        else:
            axes[i].set_title(title)
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    # Convert one-hot to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        y_true, y_pred,
        target_names=class_names or [str(i) for i in range(10)],
        digits=4
    ))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("="*80)
