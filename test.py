import configs
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(configs.img_height, configs.img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, eval_gen):
    """ Evaluate given model and print results.
    Show validation loss and accuracy, classification report and
    confusion matrix.

    Args:
        model (model): model to evaluate
        eval_gen (ImageDataGenerator): evaluation generator
    """
    # Evaluate the model
    eval_gen.reset()
    score = model.evaluate(eval_gen, verbose=0)
    print('\nLoss:', score[0])
    print('Accuracy:', score[1])

    # Confusion Matrix (validation subset)
    eval_gen.reset()
    pred = model.predict(eval_gen, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = (eval_gen.class_indices)
    target_names = labels.keys()

    # Plot statistics
    print(classification_report(eval_gen.classes, predicted_class_indices, target_names=target_names))

    cf_matrix = confusion_matrix(np.array(eval_gen.classes), predicted_class_indices)
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cf_matrix, annot=True, cmap='PuRd', cbar=False, square=True, xticklabels=target_names,
                yticklabels=target_names)
    plt.show()


evaluate_model(model, validation_generator)

# Simple way to evaluate

# load the model

from tensorflow import keras

test_batch_size = 256

X_test = tf.keras.utils.image_dataset_from_directory(
    configs.test_folder,
    batch_size=test_batch_size, #  batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    # seed=123,
)
model = keras.models.load_model('newest_model.h5')
loss, acc = model.evaluate(X_test)
print("Accuracy", acc)

