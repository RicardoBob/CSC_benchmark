from tkinter import Y
import certifi
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby

print(tf.__version__)
print(tf.keras.__version__)

# for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91190530)

# for an easy reset backend session state
tf.keras.backend.clear_session()


def load_data():
    # Get data and datasets
    config = tfds.download.DownloadConfig(verify_ssl=False)
    (train, test), info = tfds.load('ucf101', split=[
        'train', 'test'], with_info=True, download_and_prepare_kwargs={"download_config": config})

    # train, test = dataset
    # Y_train, X_train = list(map(lambda obj: (obj['label'], obj['video']), train))

    x_train, y_train = [], []
    for obj in train:
        x_train.append(obj['video'])
        y_train.append(obj['label'].numpy())

    x_test, y_test = [], []
    for obj in test:
        x_test.append(obj['video'])
        y_test.append(obj['label'].numpy())

    return (x_train, y_train), (x_test, y_test)


def prepare_data_rnn():
    pass


def prepare_data_cnn():
    pass


ucf101 = tfds.builder('ucf101')

ds = tfds.load(name="ucf101", split='train')
