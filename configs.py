import tensorflow as tf
import random

"""
Setting basic configuration for this project
mainly including:
    1.
"""
import random
PROJECT_NAME = 'cnn_model'
TEAM_NAME = 'unicorn_upc_dl'
train_folder = './data_resized/train'
val_folder = './data_resized/test'
test_folder = './data_resized/val'
seed = 168
img_height, img_width, n_channels = 256, 256, 3

augment_config = {
    'augmentation': False,
    'random_ratio': 1.0,
    'img_resize': 200,
    'img_crop': 0.2,
    'img_factor': 0.2,
    'convert_gray': True
}

input_shape = (img_height, img_width, n_channels)

data_columns = ['Image file', 'Medium', 'Museum', 'Museum-based instance ID', 'Subset',
                'Width', 'Height', 'Product size', 'Aspect ratio']

data_classes = ['Oil on canvas', 'Graphite', 'Glass', 'Limestone', 'Bronze',
                'Ceramic', 'Polychromed wood', 'Faience', 'Wood', 'Gold', 'Marble',
                'Ivory', 'Silver', 'Etching', 'Iron', 'Engraving', 'Steel',
                'Woodblock', 'Silk and metal thread', 'Lithograph',
                'Woven fabric ', 'Porcelain', 'Pen and brown ink', 'Woodcut',
                'Wood engraving', 'Hand-colored engraving', 'Clay',
                'Hand-colored etching', 'Albumen photograph']

# initializer = tf.keras.initializers.LecunNormal()

wandb_config = {
    "project_name": "cnn_aspect_ratio",
    "architecture": 'CNN',
    "epochs": 100,
    "batch_size": 128,
    'weight_decay': 0,
    'drop_rate': 0.5,
    # "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    "learning_rate": 0.0001,
    "epsilon": 1e-7,
    "amsgrad": False,
    "momentum": 0.0,
    "nesterov": False,
    "activation": 'selu',  # 'selu', 'leaky_relu'(small leak:0.01, large leak:0.2), 'gelu',
    "initialization": "he_normal",
    "optimizer": 'adam',
    # "dropout": random.uniform(0.01, 0.80),
    "normalization": True,
    "early_stopping": True,
    "augment": False,
    'initialization': 'he_normal'
    }

wandb_config.update(augment_config)

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'validation_loss'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs': {'values': [10, 20, 50]},
        'learning_rate': {'max': 0.1, 'min': 0.00001}
     }

}


