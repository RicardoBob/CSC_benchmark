# -*- coding: utf-8 -*-
"""cifar100.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RztfTSW_kWs0hbwf3ODbg9a80sry7SuM
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


def load_data(mode):

    if mode == "coarse":
        labels = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices',
                  'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                  'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
                  'trees', 'vehicles_1', 'vehicles_2']

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
            label_mode="coarse")

    elif mode == "fine":
        labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
                  'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
                  'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
                  'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                  'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                  'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                  'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                  'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                  'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
            label_mode="fine")

    return (x_train, y_train), (x_test, y_test), labels

# Matplotlib it (a set of 9 figures)


def visualize_data(x_train, y_train, classes):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]))
        x = int(y_train[i])
        label = classes[x]
        plt.xlabel(label)
    plt.show()

# Pre-processing


def prepare_data_cnn():
    (x_train, y_train), (x_test, y_test), classes = load_data("coarse")
    visualize_data(x_train, y_train, classes)
    # normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    return x_train, y_train, x_test, y_test, classes


def prepare_data_mlp():
    (x_train, y_train), (x_test, y_test), classes = load_data("coarse")
    visualize_data(x_train, y_train, classes)
    # normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
    return x_train, y_train, x_test, y_test, classes


############################################################################################################
#################################              CNN              ############################################
############################################################################################################

# CNN using the sequential API


def create_cnn(num_classes):
    model = tf.keras.Sequential()
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(16, (4, 4), (1, 1), padding='same',
                                     activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(
        16, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(32, (4, 4), (1, 1), padding='same',
                                     activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(
        32, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(
        64, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        64, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    # microarchitecture
    model.add(tf.keras.layers.Conv2D(
        128, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        128, (4, 4), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    # bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # printing a summary of the model structure
    model.summary()
    return model

    # Compile and fit the CNN


def compile_and_fit_cnn(model, X, y, batch_size, epochs, n_folds):
    # sparse_categorical_crossentropy so that we do not need to one hot encode labels
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    # k-fold cross validation
    # cross validation
    histories = list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_idx, val_idx in kfold.split(X):
        x_train = X[train_idx]
        y_train = y[train_idx]
        x_val = X[val_idx]
        y_val = y[val_idx]
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            shuffle=True)
        histories.append(history)

    return model, histories


############################################################################################################
#################################              MLP              ############################################
############################################################################################################

#layers [(neurons,activation)]
def create_mlp(num_classes, layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(3072)))
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
    num_classes = 20
    batch_size = [2**i for i in range(4, 9)]
    epochs = 2
    n_folds = 5
    # iterator for activations
    A = ["relu"]
    # iterator for number of neurons
    N = [2**i for i in range(4, 9)]

    # load data

    if model == 'cnn':
        epochs = 20
        batch = 64
        # load data
        x_train, y_train, x_test, y_test, classes = prepare_data_cnn()

        # create the model
        cnn_model = create_cnn(num_classes)

        # compile and fit model
        cnn_model, history = compile_and_fit_cnn(cnn_model, x_train, y_train,
                                                 batch, epochs, n_folds)

        # Evaluate trained model
        score = cnn_model.evaluate(x_test, y_test)

        print('Evaluation Loss:', score[0])
        print('Evaluation Accuracy:', score[1])

        # plot learning curves
        filename = "cnn_"+str(epochs)+"_"+str(batch)+"_" + "loss_" + \
            "{:.3f}".format(score[0]) + "_" + \
            "acc_"+"{:.3f}".format(score[1])
        plot_learning_curves(history, epochs)
        plt.savefig(filename+".png")

    elif model == 'mlp':

        layers_space = list(product(N, A))

        # load data
        x_train, y_train, x_test, y_test, classes = prepare_data_mlp()

        for batch in batch_size:
            for nl in range(2, 7):
                for n, a in layers_space:
                    layers = [(n, a)] * nl
                    # picture plots filename
                    filename = "mlp_"+str(epochs)+"_"+str(batch)
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


main('cnn')