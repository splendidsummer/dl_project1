"""
Setting basic configuration for this project
mainly including:
    1.
"""
import random

PROJECT_NAME = 'Fully_Connected'
TEAM_NAME = 'unicorn_upc_dl'

seed = 168
random_ratio = 0.2
img_resize = 200  # 200, 128, 64, 32
img_crop = 200  # 200, 128, 64, 32
img_factor = 0.2
train_folder = './data/train'
val_folder = './data/test'
test_folder = './data/val'
img_height, img_width = 256, 256
n_channels = 3

drop_rate = 0.2


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

wandb_config ={
    "architecture": "FN",
    "epochs": 50,
    "batch_size": 32,
    'weight_decay': 0.01,
    # "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    "learning_rate": 0.0001,
    "activation": ['relu', 'selu', 'elu',  'leaky_relu', 'gelu'],  # check the implementation in tf
    "optimizer": ['adam', 'sgd'],
    # "dropout": random.uniform(0.01, 0.80),
    "normalization": True,
    "early_stopping": True
}

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'validation_loss'
		},
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}


