import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import config
from src.data_loader import load_mnist_data, preprocess_data, get_data_augmentation, create_train_val_split
from src.model import build_residual_cnn
from src.trainer import ModelTrainer
from src.explainability import GradCAM
from src.utils import plot_confusion_matrix, plot_training_history, plot_samples, print_classification_report

def create_directories():
    """Create necessary directories for saving models and results"""
    for directory in config.PATHS.values():
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully")

def main():
    """Main execution function""" 
    print("="*80)
    print("RESIDUAL CNN FOR HANDWRITTEN DIGIT RECOGNITION WITH EXPLAINABLE AI")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Step 1: Load data
    print("\n[1/7] LOADING DATA")
    print("-"*80)
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    # Step 2: Preprocess data
    print("\n[2/7] PREPROCESSING DATA")
    print("-"*80)
    (X_train, y_train), (X_test, y_test) = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Step 3: Create train/validation split
    print("\n[3/7] CREATING TRAIN/VALIDATION SPLIT")
    print("-"*80)
    (X_train_split, y_train_split), (X_val, y_val) = create_train_val_split(
        X_train, y_train, 
        validation_split=config.TRAINING_CONFIG['validation_split']
    )
    
    # Step 4: Build model
    print("\n[4/7] BUILDING RESIDUAL CNN MODEL")
    print("-"*80)
    model = build_residual_cnn(
        input_shape=config.MODEL_CONFIG['input_shape'],
        num_classes=config.MODEL_CONFIG['num_classes'],
        num_blocks=config.MODEL_CONFIG['residual_blocks']
    )
    model.summary()
    
    # Step 5: Train model
    print("\n[5/7] TRAINING MODEL")
    print("-"*80)
    trainer = ModelTrainer(model)
    datagen = get_data_augmentation()
    
    history = trainer.train(
        X_train_split, y_train_split,
        X_val, y_val,
        datagen=datagen
    )
    
    # Save training history plot
    fig = plot_training_history(history)
    plt.savefig(f"{config.PATHS['results_dir']}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {config.PATHS['results_dir']}/training_history.png")
    
    # Step 6: Evaluate on test set
    print("\n[6/7] EVALUATING ON TEST SET")
    print("-"*80)
    test_metrics = trainer.evaluate(X_test, y_test)
    
    # Get predictions for confusion matrix and classification report
    y_pred = trainer.get_predictions(X_test)
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(y_test, y_pred, class_names=[str(i) for i in range(10)])
    plt.savefig(f"{config.PATHS['results_dir']}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {config.PATHS['results_dir']}/confusion_matrix.png")
    
    # Print detailed classification report
    print_classification_report(y_test, y_pred, class_names=[str(i) for i in range(10)])
    
    # Plot sample predictions
    fig = plot_samples(X_test[:16], y_test[:16], y_pred[:16], num_samples=16)
    plt.savefig(f"{config.PATHS['results_dir']}/sample_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {config.PATHS['results_dir']}/sample_predictions.png")
    
    # Step 7: GRAD-CAM Explainability
    print("\n[7/7] GRAD-CAM EXPLAINABILITY ANALYSIS")
    print("-"*80)
    
    # Find the last convolutional layer
    conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            conv_layer_name = layer.name
            break
    
    if conv_layer_name is None:
        conv_layer_name = 'conv1'
    
    print(f"Using layer '{conv_layer_name}' for GRAD-CAM visualization")
    
    # Initialize GRAD-CAM
    gradcam = GradCAM(model, layer_name=conv_layer_name)
    
    # Get correctly and incorrectly classified samples
    predictions = trainer.get_predictions(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    correct_mask = pred_classes == true_classes
    incorrect_mask = ~correct_mask
    
    # Plot GRAD-CAM for correctly classified samples
    if np.sum(correct_mask) > 0:
        correct_indices = np.where(correct_mask)[0][:5]
        correct_images = X_test[correct_indices]
        correct_preds = predictions[correct_indices]
        
        fig = GradCAM.plot_multiple_gradcam(
            correct_images,
            correct_preds,
            class_names=[str(i) for i in range(10)],
            model_instance=gradcam,
            num_samples=5
        )
        plt.savefig(f"{config.PATHS['results_dir']}/gradcam_correct_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GRAD-CAM for correct predictions saved to {config.PATHS['results_dir']}/gradcam_correct_predictions.png")
    
    # Plot GRAD-CAM for incorrectly classified samples
    if np.sum(incorrect_mask) > 0:
        incorrect_indices = np.where(incorrect_mask)[0][:5]
        incorrect_images = X_test[incorrect_indices]
        incorrect_preds = predictions[incorrect_indices]
        
        fig = GradCAM.plot_multiple_gradcam(
            incorrect_images,
            incorrect_preds,
            class_names=[str(i) for i in range(10)],
            model_instance=gradcam,
            num_samples=5
        )
        plt.savefig(f"{config.PATHS['results_dir']}/gradcam_incorrect_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"GRAD-CAM for incorrect predictions saved to {config.PATHS['results_dir']}/gradcam_incorrect_predictions.png")
    
    # Save the trained model
    trainer.save_model(f"{config.PATHS['models_dir']}/final_model.h5")
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {config.PATHS['results_dir']}")
    print(f"Model saved to: {config.PATHS['models_dir']}")
    print("\nGenerated files:")
    print("  - training_history.png: Training and validation curves")
    print("  - confusion_matrix.png: Confusion matrix heatmap")
    print("  - sample_predictions.png: Sample predictions with confidence scores")
    print("  - gradcam_correct_predictions.png: GRAD-CAM for correct predictions")
    print("  - gradcam_incorrect_predictions.png: GRAD-CAM for incorrect predictions")
    print("  - final_model.h5: Trained model")
    print("="*80)


if __name__ == "__main__":
    main()