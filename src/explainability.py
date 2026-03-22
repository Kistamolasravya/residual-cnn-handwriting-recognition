"""
GRAD-CAM (Gradient-weighted Class Activation Mapping) for model explainability
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Map (GRAD-CAM) for explaining model predictions
    """
    
    def __init__(self, model, layer_name):
        """
        Initialize GRAD-CAM
        
        Args:
            model: Keras model
            layer_name: Name of the layer to visualize
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None
        self._build_grad_model()
    
    def _build_grad_model(self):
        """Build gradient model for computing activations and gradients"""
        layer = self.model.get_layer(self.layer_name)
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [layer.output, self.model.output]
        )
    
    def compute_gradcam(self, img, class_idx, eps=1e-8):
        """
        Compute GRAD-CAM heatmap for an image
        
        Args:
            img: Input image (single sample, shape: (height, width, channels))
            class_idx: Class index for which to compute gradients
            eps: Small epsilon to avoid division by zero
        
        Returns:
            heatmap: GRAD-CAM heatmap
        """
        img_tensor = tf.expand_dims(img, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            class_channel = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute weights (average pooling of gradients)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute weighted combination of activation maps
        cam = tf.reduce_sum(
            tf.multiply(weights, conv_outputs[0]), axis=-1
        )
        
        # Normalize the CAM
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + eps)
        
        return cam.numpy()
    
    def plot_gradcam(self, img, class_idx, class_name, figsize=(12, 4)):
        """
        Plot original image, GRAD-CAM heatmap, and overlay
        
        Args:
            img: Input image
            class_idx: Predicted class index
            class_name: Name of predicted class
            figsize: Figure size
        """
        # Compute GRAD-CAM
        heatmap = self.compute_gradcam(img, class_idx)
        
        # Resize heatmap to original image size
        heatmap = tf.image.resize(
            tf.expand_dims(tf.expand_dims(heatmap, axis=0), axis=-1),
            (img.shape[0], img.shape[1])
        ).numpy().squeeze()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot original image
        axes[0].imshow(img.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot heatmap
        im = axes[1].imshow(heatmap, cmap='hot')
        axes[1].set_title('GRAD-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot overlay
        axes[2].imshow(img.squeeze(), cmap='gray')
        axes[2].imshow(heatmap, cmap='hot', alpha=0.5)
        axes[2].set_title(f'Overlay\nPredicted: {class_name}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_multiple_gradcam(images, predictions, class_names, model_instance, num_samples=5):
        """
        Plot GRAD-CAM for multiple images
        
        Args:
            images: Array of images
            predictions: Model predictions
            class_names: List of class names (0-9 for digits)
            model_instance: GradCAM instance
            num_samples: Number of samples to plot
        """
        num_show = min(num_samples, len(images))
        
        fig = plt.figure(figsize=(15, 3 * num_show))
        
        for i in range(num_show):
            pred_class = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            
            # Original image
            ax1 = plt.subplot(num_show, 3, i * 3 + 1)
            ax1.imshow(images[i].squeeze(), cmap='gray')
            ax1.set_title(f'Original\nDigit: {pred_class}')
            ax1.axis('off')
            
            # GRAD-CAM
            heatmap = model_instance.compute_gradcam(images[i], pred_class)
            heatmap = tf.image.resize(
                tf.expand_dims(tf.expand_dims(heatmap, axis=0), axis=-1),
                (images[i].shape[0], images[i].shape[1])
            ).numpy().squeeze()
            
            ax2 = plt.subplot(num_show, 3, i * 3 + 2)
            im = ax2.imshow(heatmap, cmap='hot')
            ax2.set_title('GRAD-CAM\nHeatmap')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            # Overlay
            ax3 = plt.subplot(num_show, 3, i * 3 + 3)
            ax3.imshow(images[i].squeeze(), cmap='gray')
            ax3.imshow(heatmap, cmap='hot', alpha=0.5)
            ax3.set_title(f'Overlay\nConfidence: {confidence:.2%}')
            ax3.axis('off')
        
        plt.tight_layout()
        return fig
