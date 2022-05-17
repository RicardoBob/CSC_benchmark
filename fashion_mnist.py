#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:38:17 2022

@author: ricardo
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

print(tf.__version__)
print(tf.keras.__version__)

# for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91190530)

# for an easy reset backend session state
tf.keras.backend.clear_session()

# Loading training and the testing sets (numpy arrays)


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # we have 10 labels (0:T-shirt, 1:Trouser, ..., 9:Ankle boot)
    # each image is mapped to one single label (class names not included)
    classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
               "Bag", "Ankle boot"]
    return (x_train, y_train), (x_test, y_test), classes

# Matplotlib it (a set of 9 figures)


def visualize_data(x_train, y_train, classes):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]))
        plt.xlabel(classes[y_train[i]])
    plt.show()

# Pre-processing


def prepare_data_cnn():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    #visualize_data(x_train, y_train, classes)
    # normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, classes


def prepare_data_mlp():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    #visualize_data(x_train, y_train, classes)
    # normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    return x_train, y_train, x_test, y_test, classes


def prepare_data_rnn():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    #visualize_data(x_train, y_train, classes)

############################################################################################################
#################################              CNN              ############################################
############################################################################################################

# CNN using the sequential API


def create_cnn(num_classes):
    model = tf.keras.Sequential()
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same',
                                     activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    # output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # printing a summary of the model structure
    model.summary()
    return model

    # Compile and fit the CNN


def compile_and_fit_cnn(model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation):
    # sparse_categorical_crossentropy so that we do not need to one hot encode labels
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # fit with/without data augmentation
    if not apply_data_augmentation:
        print('No data augmentation')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)
    else:
        print('Using data augmentation')
        # preprocessing and realtime data augmentation with ImageDataGenerator
        datagen = ImageDataGenerator(
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=90,
            zoom_range=0.,  # set range for random zoom
            horizontal_flip=False,  # randomly horizontally flip images
            vertical_flip=True,  # randomly vertically flip images
            rescale=None,  # rescaling factor (applied before any other transf)
            preprocessing_function=None  # function applied on each input
        )
        # compute quantities required for feature-wise normalization
        datagen.fit(x_train)
        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      workers=1)
    return model, history


############################################################################################################
#################################              MLP              ############################################
############################################################################################################

#layers [(neurons,activation)]
def create_mlp(num_classes, layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(784)))
    # add hidden layers
    for neurons, activation in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def compile_and_fit_mlp(model, X, y, batch_size, epochs, n_folds):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # cross validation
    histories = list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_idx, val_idx in kfold.split(X):
        x_train = X[train_idx]
        y_train = y[train_idx]
        x_val = X[val_idx]
        y_val = y[val_idx]
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(x_val, y_val), shuffle=True)
        histories.append(history)

    return model, histories


############################################################################################################
#################################              MAIN             ############################################
############################################################################################################

# Vizualizing Learning Curves


def plot_learning_curves(history, epochs):
    #accuracies and losses
    acc = [0] * epochs
    val_acc = [0] * epochs
    loss = [0] * epochs
    val_loss = [0] * epochs
    for h in history:
        acc = np.add(acc, h.history['accuracy'])
        val_acc = np.add(val_acc, h.history['val_accuracy'])
        loss = np.add(loss, h.history['loss'])
        val_loss = np.add(val_loss, h.history['val_loss'])
    for i in range(epochs):
        acc[i] /= len(history)
        val_acc[i] /= len(history)
        loss[i] /= len(history)
        val_loss[i] /= len(history)
    epochs_range = range(epochs)
    # creating figure
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training/Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss')


def main(model):
    # Main Execution (Run it in Colab)
    num_classes = 10
    batch_size = [2**i for i in range(5, 11)]
    epochs = 2
    n_folds = 5
    apply_data_augmentation = False
    # iterator for activations
    A = ["relu", "tanh"]
    # iterator for number of neurons
    N = [2**i for i in range(4, 8)]

    # load data

    if model == 'cnn':
        # load data
        x_train, y_train, x_test, y_test, classes = prepare_data_cnn()

        # create the model
        cnn_model = create_cnn(num_classes)

        # compile and fit model
        cnn_model, history = compile_and_fit_cnn(cnn_model, x_train, y_train, x_test, y_test,
                                                 batch_size, epochs, apply_data_augmentation)

        # Evaluate trained model
        score = cnn_model.evaluate(x_test, y_test)
        print('Evaluation Loss:', score[0])
        print('Evaluation Accuracy:', score[1])

    elif model == 'mlp':

        layers_space = list(product(N, A))

        # load data
        x_train, y_train, x_test, y_test, classes = prepare_data_mlp()

        for batch in batch_size:
            for nl in range(1, 7):
                for n, a in layers_space:
                    layers = [(n, a)] * nl
                    # picture plots filename
                    filename = str(epochs)+"_"+str(batch)
                    for n, a in layers:
                        filename += "_" + "(" + str(n) + "-" + str(a) + ")"
                    filename += "_"

                    # create the model
                    mlp_model = create_mlp(num_classes, layers)

                    # compile and fit model
                    mlp_model, history = compile_and_fit_mlp(
                        mlp_model, x_train, y_train, batch, epochs, n_folds)

                    # Evaluate trained model
                    score = mlp_model.evaluate(x_test, y_test)

                    filename += "loss_" + \
                        "{:.3f}".format(score[0]) + "_" + \
                        "acc_"+"{:.3f}".format(score[1])

                    print('Evaluation Loss:', score[0])
                    print('Evaluation Accuracy:', score[1])

                    # plot learning curves
                    plot_learning_curves(history, epochs)
                    plt.savefig(filename+".png")


main('mlp')
