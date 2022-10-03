import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json

data_columns = ['Image file', 'Medium', 'Museum', 'Museum-based instance ID', 'Subset',
                'Width', 'Height', 'Product size', 'Aspect ratio']

data_classes = ['Oil on canvas', 'Graphite', 'Glass', 'Limestone', 'Bronze',
                'Ceramic', 'Polychromed wood', 'Faience', 'Wood', 'Gold', 'Marble',
                'Ivory', 'Silver', 'Etching', 'Iron', 'Engraving', 'Steel',
                'Woodblock', 'Silk and metal thread', 'Lithograph',
                'Woven fabric ', 'Porcelain', 'Pen and brown ink', 'Woodcut',
                'Wood engraving', 'Hand-colored engraving', 'Clay',
                'Hand-colored etching', 'Albumen photograph']


def read_split_data(csv_data_file, csv_label_file):
    """
    :param csv_file: csv file in MAMe_metadata folder:
        1. MAME_dataset6
        2. MAME_labels
    :return:
    Organized file path in dict format:
        dict key: image class
        dict value: all the image data paths corresponding to the key
    save the processed data into pkl file as well
    """
    data = pd.read_csv(csv_data_file)
    labels = pd.read_csv(csv_label_file)

    return None


def read_csv():
    pass


def explore_data():
    pass

import tensorflow as tf
base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = (32,32,3))