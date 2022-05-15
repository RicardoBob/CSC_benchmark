#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:46:50 2022

@author: ricardo
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)

#for replicability purposes
tf.random.set_seed(91195003)
np.random.seed(91190530)

#for an easy reset backend session state
tf.keras.backend.clear_session()

#Loading training and the testing sets (numpy arrays)
def load_data():
    (_, y_train_fine),(_, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    (x_train, y_train_coarse), (x_test, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    y_train = tf.concat([y_train_coarse,y_train_fine], 1)
    y_test = tf.concat([y_test_coarse,y_test_fine], 1)
    print(y_test[10].numpy())
    
    coarse_labels = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 
                    'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 
                    'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 
                    'trees', 'vehicles_1', 'vehicles_2']
    
    fine_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
                  'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 
                  'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 
                  'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
                  'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                  'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
                  'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 
                  'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 
                  'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'] 
    

    return (x_train, y_train), (x_test, y_test), coarse_labels, fine_labels
    
#Matplotlib it (a set of 9 figures)
def visualize_data(x_train, y_train, fine_labels, coarse_labels):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]))
        x,y = y_train[i]
        label = coarse_labels[x] + " and " + fine_labels[y]
        plt.xlabel(label)
    plt.show()

#Pre-processing
def prepare_data_cnn():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    visualize_data(x_train, y_train, classes)
    #normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, classes

def prepare_data_mlp():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    visualize_data(x_train, y_train, classes)
    #normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    return x_train, y_train, x_test, y_test, classes

def prepare_data_rnn():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    visualize_data(x_train, y_train, classes)

############################################################################################################
#################################              CNN              ############################################
############################################################################################################

#CNN using the sequential API
def create_cnn(num_classes):
    model = tf.keras.Sequential()
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same',
                                     activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    #output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #printing a summary of the model structure
    model.summary()
    return model

    #Compile and fit the CNN
def compile_and_fit_cnn(model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation):
    #sparse_categorical_crossentropy so that we do not need to one hot encode labels
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    #fit with/without data augmentation
    if not apply_data_augmentation:
        print('No data augmentation')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True)
    else:
        print('Using data augmentation')
        #preprocessing and realtime data augmentation with ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range = 90, #randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0., #set range for random zoom
            horizontal_flip = False, #randomly horizontally flip images
            vertical_flip = True, #randomly vertically flip images
            rescale = None, #rescaling factor (applied before any other transf)
            preprocessing_function = None #function applied on each input
            )
        #compute quantities required for feature-wise normalization
        datagen.fit(x_train)
        #fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
                                      epochs = epochs,
                                      validation_data = (x_test, y_test),
                                      workers = 1)
    return model, history



############################################################################################################
#################################              MLP              ############################################
############################################################################################################

def create_mlp(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(None,784)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def compile_and_fit_mlp(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),shuffle=True)
    return model, history



############################################################################################################
#################################              RNN              ############################################
############################################################################################################

def create_rnn(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same',
                                     activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()
    return model

def compile_and_fit_rnn(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=epochs, batch_size=batch_size, shuffle=False)
    return model, history

############################################################################################################
#################################              MAIN             ############################################
############################################################################################################

#Vizualizing Learning Curves
def plot_learning_curves(history, epochs):
    #accuracies and losses
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    #creating figure
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
    plt.show()


def main(model):
    #Main Execution (Run it in Colab)
    num_classes = 10
    batch_size = 128
    epochs = 10
    apply_data_augmentation = False
    num_predictions = 20

    #load data
    
    if model == 'cnn':
        #load data
        x_train, y_train, x_test, y_test, classes = prepare_data_cnn()
        
        #create the model
        cnn_model = create_cnn(num_classes)

        #compile and fit model
        cnn_model, history = compile_and_fit_cnn(cnn_model, x_train, y_train, x_test, y_test,
                                             batch_size, epochs, apply_data_augmentation)

        #Evaluate trained model
        score = cnn_model.evaluate(x_test, y_test)
        print('Evaluation Loss:', score[0])
        print('Evaluation Accuracy:', score[1])
        
    elif model == 'mlp':
        #load data
        x_train, y_train, x_test, y_test, classes = prepare_data_mlp()
        
        #create the model
        mlp_model = create_mlp(num_classes)
        
        #compile and fit model
        mlp_model, history = compile_and_fit_mlp(mlp_model, x_train, y_train, x_test, y_test, batch_size, epochs)
        
        #Evaluate trained model
        score = mlp_model.evaluate(x_train,y_train)
        print('Evaluation Loss:', score[0])
        print('Evaluation Accuracy:', score[1])
        
    else:
        #load data
        x_train, y_train, x_test, y_test, classes = prepare_data_cnn()
        
        #create the model
        rnn_model = create_rnn(num_classes)
        
        #compile and fit model
        rnn_model, history = compile_and_fit_rnn(rnn_model, x_train, y_train, x_test, y_test, batch_size, epochs)
        
        #Evaluate trained model
        score = rnn_model.evaluate(x_train,y_train)
        print('Evaluation Loss:', score[0])
        print('Evaluation Accuracy:', score[1])

        
    #plot learning curves    
    plot_learning_curves(history, epochs)
    
(x_train, y_train), (x_test, y_test), coarse_labels, fine_labels = load_data()
visualize_data(x_train, y_train, fine_labels, coarse_labels)
