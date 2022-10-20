import configs
import tensorflow as tf
from keras import layers, Sequential
from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,\
    UpSampling2D, Add
from keras.layers import RandomRotation, RandomZoom, CenterCrop, \
    RandomContrast, RandomCrop, RandomTranslation, Rescaling, Resizing
from functools import partial
from customized import *

data_augmentation = Sequential(
  [
    layers.Resizing(configs.augment_config['img_resize'], configs.augment_config['img_resize']),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal_and_vertical"),

    layers.RandomRotation(configs.augment_config['random_ratio']),
    layers.CenterCrop(configs.augment_config['img_crop'], configs.augment_config['img_crop']),
    layers.RandomZoom(configs.augment_config['random_ratio']),
    # layers.RandomBrightness(configs.img_factor),
    layers.RandomContrast(configs.augment_config['img_factor']),
    layers.RandomTranslation(configs.augment_config['img_factor'], configs.augment_config['img_factor']),
    RandomGray(configs.augment_config['random_ratio']),
  ]
)


# Fully-connected layers architecture
def fc_model1():
    """
    network architecture summary:
        Purely fully-connected layer
        1st layer 2048 hidden units;
        2nd layer 512 hidden units;
    :return:
    """

    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),

        # data_augmentation,

        layers.Resizing(64, 64),
        layers.Rescaling(1. / 255),

        layers.Flatten(),

        layers.Dense(2048, activation='relu'),
        layers.Dense(512, activation='relu'),

        # layers.Dense(128, activation='relu'),
        layers.Dropout(configs.wandb_config['drop_rate']),

        layers.Dense(len(configs.data_classes))])

    return model
