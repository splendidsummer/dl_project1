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


###############################################
# Building convolution models without BN
###############################################
def cnn4(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(),
        #
        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        # # layers.BatchNormalization(),
        layers.MaxPooling2D(),


        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model2(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model5(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model5_bottleneck():
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 5, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr5_activation_bottleneck(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 5, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr5_activation_3333(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model

def cbr5_leakyrelu_bottleneck(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', kernel_initializer=config.initialization),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 5, padding='same', kernel_initializer=config.initialization),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', kernel_initializer=config.initialization),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', kernel_initializer=config.initialization),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', kernel_initializer=config.initialization),
        layers.LeakyReLU(alpha=0.01),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model5_bottleneck_7533(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 5, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(configs.wandb_config['drop_rate']),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model5_bottleneck_7333_2(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config['drop_rate']),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def cbr_model5_bottleneck_7333(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 7, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 5, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(configs.wandb_config['drop_rate']),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb_model3(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),


        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(config.drop_rate),
        layers.Flatten(),

        layers.Dense(512, activation=config.activation),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb_model4(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),


        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(config.drop_rate),
        layers.Flatten(),

        layers.Dense(512, activation=config.activation),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb_model5_bottleneck(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),


        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(config.drop_rate),
        layers.Flatten(),

        layers.Dense(512, activation=config.activation),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb5_bottleneck_dense2(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(config.drop_rate),
        layers.Flatten(),

        layers.Dense(1024, activation=config.activation),
        layers.Dropout(config.drop_rate),
        layers.Dense(512, activation=config.activation),

        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb5_bottleneck_dense2_dropout(config):
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(16, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 1, activation=config.activation, kernel_initializer=config.initialization),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(config.drop_rate),
        layers.Flatten(),

        layers.Dense(1024, activation=config.activation),
        layers.Dropout(config.drop_rate),
        layers.Dense(512, activation=config.activation),

        layers.Dense(len(configs.data_classes)),
    ])

    return model


def crb_model2():
    """
    network architecture summary:
        1.
    :return:
    """
    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        # data_augmentation,
        # layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),

        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(configs.wandb_config.drop_rate),
        layers.Flatten(),

        layers.Dense(512, activation='relu'),
        layers.Dense(len(configs.data_classes)),
    ])

    return model


def model_conv32():
    """
    network architecture summary:
        1.
    :return:
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', padding='same',
                     input_shape=configs.input_shape, data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(configs.data_classes)))

    # Print model summary
    # print(model.summary())

    return model


def skip_model():
    return None


tf.random.set_seed(42)  # extra code – ensures reproducibility
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10)
])


def define_skip_model():
    input_net = layers.Input((32, 32, 3))

    ## Encoder starts
    conv1 = Conv2D(32, 3, strides=(2, 2), activation='relu', padding='same')(input_net)
    conv2 = Conv2D(64, 3, strides=(2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same')(conv2)

    conv4 = Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same')(conv3)

    ## And now the decoder
    up1 = Conv2D(128, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
    merge1 = layers.concatenate([conv3, up1], axis=3)
    up2 = Conv2D(64, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(merge1))
    merge2 = layers.concatenate([conv2, up2], axis=3)
    up3 = Conv2D(32, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(merge2))
    merge3 = layers.concatenate([conv1, up3], axis=3)

    up4 = Conv2D(32, 3, padding='same')(UpSampling2D(size=(2, 2))(merge3))

    output_net = Conv2D(3, 3, padding='same')(up4)

    model = Model(inputs=input_net, outputs=output_net)

    return model


X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
   def __init__(self):
       super(Linear).__init__()
       self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer())

   def call(self, input):
       output = self.dense(input)
       return output


def _skip_block(inputs, num_filter):
    x = inputs
    if inputs.shape[-1] != num_filter:
        x = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
    # x = layers.Conv2D(num_filter, 1,  padding='same', kernel_initializer='he_normal')(inputs)
    y = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
    y = layers.Conv2D(num_filter, 3, padding='same', kernel_initializer='he_normal')(y)
    y = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(y)

    out = Add()([x, y])
    return out


def _skip_block2(inputs, num_filter):
    x = inputs
    if inputs.shape[-1] != num_filter:
        x = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
    # x = layers.Conv2D(num_filter, 1,  padding='same', kernel_initializer='he_normal')(inputs)
    y = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(inputs)
    y = layers.Conv2D(num_filter, 3, padding='same', kernel_initializer='he_normal')(y)
    y = layers.Conv2D(num_filter, 1, padding='same', kernel_initializer='he_normal')(y)

    out = Add()([x, y])
    return out


class VannilaSikpModel(tf.keras.Model):
    def __init__(self, config, num_classes=29):
        super(VannilaSikpModel, self).__init__()
        self.conv0 = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 3))
        self.conv11 = layers.Conv2D(64, 1, padding='same', kernel_initializer='he_normal')
        self.conv12 = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')
        self.conv13 = layers.Conv2D(64, 1, padding='same', kernel_initializer='he_normal')
        self.maxpool1 = layers.MaxPooling2D()
        self.add1 = layers.Add()

        self.conv20 = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')
        self.conv21 = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')
        self.conv22 = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')
        self.conv23 = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')
        self.maxpool2 = layers.MaxPooling2D()
        self.add2 = layers.Add()

        self.dropout = layers.Dropout(configs.wandb_config['drop_rate'])
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')

    def call(self, inputs):
        x = self.conv0(inputs)
        out = self.conv11(x)
        out = self.conv12(out)
        out = self.conv13(out)
        out = self.add1([x, out])
        out = self.maxpool1(out)

        x = self.conv20(out)

        out = self.conv21(x)
        out = self.conv22(out)
        out = self.conv23(out)
        out = self.add2([x, out])
        out = self.maxpool2(out)

        out = self.flatten(self.dropout(out))
        out = self.dense(out)

        return out


#     def identity_block(input_tensor, kernel_size, filters, stage, block):
#         """The identity block is the block that has no conv layer at shortcut.
#         # Arguments
#             input_tensor: input tensor
#             kernel_size: default 3, the kernel size of
#                 middle conv layer at main path
#             filters: list of integers, the filters of 3 conv layer at main path
#             stage: integer, current stage label, used for generating layer names
#             block: 'a','b'..., current block label, used for generating layer names
#         # Returns
#             Output tensor for the block.
#         """
#         filters1, filters2, filters3 = filters
#         if backend.image_data_format() == 'channels_last':
#             bn_axis = 3
#         else:
#             bn_axis = 1
#         conv_name_base = 'res' + str(stage) + block + '_branch'
#         bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#         x = layers.Conv2D(filters1, (1, 1),
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2a')(input_tensor)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters2, kernel_size,
#                           padding='same',
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2b')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#         x = layers.Activation('relu')(x)
#
#         x = layers.Conv2D(filters3, (1, 1),
#                           kernel_initializer='he_normal',
#                           name=conv_name_base + '2c')(x)
#         x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#         x = layers.add([x, input_tensor])
#         x = layers.Activation('relu')(x)
#         return x
#
