import tensorflow as tf
from utils import *
import configs
from models import *
from keras import layers
from keras import layers, Sequential
from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,\
    UpSampling2D, Add

import os, sys, time, tqdm, datetime
from wandb.keras import WandbCallback
from keras.optimizers import SGD, Adam
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

# wandb.login()

set_seed(configs.seed)
print(tf.config.list_physical_devices('GPU'))

# initialize wandb logging to your project
wandb.init(
    project=configs.PROJECT_NAME,
    dir = '/root/autodl-tmp/dl_project12/wandb_logs',
    entity=configs.TEAM_NAME,
    config=configs.wandb_config,
    sync_tensorboard=True,
    ####
)
config = wandb.config

lr = config.learning_rate
weight_decay = config.weight_decay
early_stopping = config.early_stopping
activation = config.activation
normalization = config.normalization
data_augmentation = config.data_augmentation

X_train = tf.keras.utils.image_dataset_from_directory(
    configs.train_folder,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=True,
    seed=configs.seed
)
# class_names = X_train.class_names
# print(class_names)

X_val = tf.keras.utils.image_dataset_from_directory(
    configs.val_folder,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    # seed=123,
)

AUTOTUNE = tf.data.AUTOTUNE
X_train = X_train.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)

print('building model')
# Selecting a model from model library
# model = fc_model()

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', padding='same',
                 input_shape=(256, 256, 3), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(29))

optimizerW = tfa.optimizers.AdamW(
    weight_decay=config.weight_decay,
    learning_rate=config.learning_rate,

)
# optimizer = Adam(lr=config.learning_rate)

optimizer=Adam(lr=0.001, epsilon=0.1, amsgrad=True)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer, metrics=["accuracy"])
# Use early stopping
early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)
wandb_callback = WandbCallback()

# optimizer = Adam(lr=0.001, epsilon=0.1, amsgrad=True) do we need to change the epsilon?
# step = tf.Variable(0, trainable=False)
# schedule = tf.optimizers.schedules.PiecewiseConstantDecay()

# Train the model
t0 = time.time()
epochs = 100

t0 = time.time()

print('training model')

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=config.epochs,
                    # callbacks=[early_callback, wandb_callback],  # other callback?
                    )

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

# Saving model
now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
model.save(now + '.h5')






