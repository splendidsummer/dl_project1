import tensorflow as tf
from utils import *
from cnn_models import *
import configs
from models import *
from keras import layers
from keras import layers, Sequential
from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,\
    UpSampling2D, Add

import os, sys, time, tqdm, datetime
from keras.optimizers import SGD, Adam
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbCallback

model = cbr_model5_bottleneck()

sweep_id = wandb.sweep(sweep=configs.sweep_configuration, project="sweeps_projects")


def onerun():
    wandb.init(
        project='cbr_sweep',
    )
    config = wandb.config
    lr = config.learning_rate
    epochs = config.epochs
    batch_size = config.batch_size
    X_train = tf.keras.utils.image_dataset_from_directory(configs.train_folder,
    batch_size=batch_size,
    shuffle=True,
    seed=configs.seed)

    X_val = tf.keras.utils.image_dataset_from_directory(configs.val_folder,
    batch_size=batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,)

    optimizer = Adam(lr=lr)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer, metrics=["accuracy"])
    # Use early stopping
    early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)
    wandb_callback = WandbCallback()

    print('Starting training!!')

    history = model.fit(X_train,
                        validation_data=X_val,
                        epochs=epochs,
                        callbacks=[early_callback, wandb_callback],
                        )


wandb.agent(sweep_id=sweep_id, function=onerun)

