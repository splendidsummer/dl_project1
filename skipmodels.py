import configs
import tensorflow as tf
from keras import layers, Sequential
from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,\
    UpSampling2D, Add, Activation
from keras.layers import RandomRotation, RandomZoom, CenterCrop, \
    RandomContrast, RandomCrop, RandomTranslation, Rescaling, Resizing
from functools import partial
from customized import *


# Basic resnet block
############################################
# model class
############################################
class SikpConnectionModel(Model):
    def __init__(self, num_classes=29):
        super(SikpConnectionModel, self).__init__()
        self.conv_block1 = self._cbr_block(16, 3)

    def _cbr_block(self, num_filter, kernel_size=3):
        cbr_block = Sequential([
            layers.Conv2D(num_filter, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),
        ])
        return cbr_block

    def _basic_skip_block(self, inputs, num_filter, kernel_size=3):
        x = inputs
        y = 

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x




