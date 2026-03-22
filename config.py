# Configuration settings for the project

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 10,
    'residual_blocks': [2, 2, 2, 2],
    'initial_filters': 64,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.1,
    'early_stopping_patience': 10,
}

# Data Configuration
DATA_CONFIG = {
    'dataset_name': 'mnist',
    'data_path': './data',
    'normalize': True,
    'augmentation': True,
}

# Evaluation Configuration
EVAL_CONFIG = {
    'test_split': 0.2,
    'save_confusion_matrix': True,
    'plot_misclassified': True,
}

# Explainability Configuration
EXPLAINABILITY_CONFIG = {
    'method': 'gradcam',
    'layer_name': 'conv5_block3_out',
    'num_samples': 5,
}

# Paths
PATHS = {
    'models_dir': './models',
    'data_dir': './data',
    'logs_dir': './logs',
    'results_dir': './results',
}
