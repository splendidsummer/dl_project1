from keras import Sequential
from keras import layers
import configs

data_augmentation = Sequential(
  [
    layers.Resizing(configs.img_resize, configs.img_resize),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomFlip("horizontal_and_vertical"),

    layers.RandomRotation(configs.random_ratio),
    layers.CenterCrop(configs.img_crop, configs.img_crop),
    layers.RandomZoom(configs.random_ratio),
    layers.RandomBrightness(configs.img_factor),
    layers.RandomContrast(configs.img_factor),
    layers.RandomTranslation(configs.img_factor, configs.img_factor),
  ]
)


# Fully-connected layers architecture
def fc_model():
    """
    network architecture summary:
        Purely fully-connected layer
        1st layer 2048 hidden units;
        2nd layer 512 hidden units;
    :return:
    """

    model = Sequential([
        layers.InputLayer(input_shape=configs.input_shape),
        data_augmentation,
        layers.Resizing(configs.img_resize, configs.img_resize),
        layers.Rescaling(1. / 255),
        layers.Dense(2048, activation='relu'),
        layers.Dense(512, activation='relu'),
        # layers.Dense(128, activation='relu'),
        layers.Dropout(configs.drop_rate),
        layers.Dense(len(configs.data_classes))])

    # Print model summary
    # print(model.summary())

    return model


if __name__ == '__main__':
    # model = fc_model()
    #
    # model = Sequential([
    #     layers.InputLayer(input_shape=configs.input_shape),
    #     data_augmentation,
    #     layers.Resizing(configs.img_resize, configs.img_resize),
    #     layers.Flatten(),
    #     # layers.Rescaling(1. / 255),
    #     layers.Dense(2048, activation='relu'),
    #     layers.Dense(512, activation='relu'),
    #     # layers.Dense(128, activation='relu'),
    #     layers.Dropout(configs.drop_rate),
    #     layers.Dense(len(configs.data_classes))])
    #
    # model.summary()
    #
    import datetime, time

    now = datetime.datetime.now()
    print(now)
    # now = time.strptime(now)

    # dt = datetime.datetime.strptime(this_time, '%Y%m%d%H%M%S')
    # dt = datetime.datetime.strptime(now, '%Y%m%d%H%M%S')

    # print(dt)
    # 2010-03-04 08:28:35

# Normalization of input images
# normalization_layer = layers.Rescaling(1./255)
# normalized_train = X_train.map(lambda x, y: (normalization_layer(x), y))
# normalized_val = X_val.map(lambda x, y: (normalization_layer(x), y))
# normalized_test = X_test.map(lambda x, y: (normalization_layer(x), y))



tf.keras.