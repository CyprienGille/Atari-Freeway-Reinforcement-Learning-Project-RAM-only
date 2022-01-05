"""
Script to train the artificial neural network: run it to get a trained_agent saved to the working dir.

Most interesting parameters are defined at the start of the script for easy changing.
Note: This script assumes that you already created data with the data_creation.py script, or in a similar way.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

if __name__ == "__main__":

    ######################### Constants/params ####################
    TRAIN_PROP = 0.80 # The rest will be reserved for validation
    BATCH_SIZE = 32 # not very important in our case
    EPOCHS = 300 # max number of epochs, almost never reached

    # how many epochs without improvement to wait before stopping training
    EARLY_STOPPING_PATIENCE = 30
    DROPOUT_RATE = 0.05
    ARRF_PATH = "data/arrF.npy"
    ARRL_PATH = "data/arrL.npy"

    PLOT_AFTER_TRAINING = True # plot accuracies and losses after training

    ######################### Data loading ####################
    features_array = np.load(ARRF_PATH)
    labels_array = np.load(ARRL_PATH)

    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    dataset_size = dataset.cardinality().numpy()

    dataset = dataset.shuffle(dataset_size + 1, seed=321)
    train_ds = dataset.take(int(TRAIN_PROP*dataset_size))
    val_ds = dataset.skip(int(TRAIN_PROP*dataset_size))
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    ######################### Model definition ####################
    model = Sequential([
        layers.Dense(128, input_shape=(128,), activation='relu', name="input"),
        layers.Dense(64, activation='relu', name="dense2"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(64, activation='relu', name="dense3"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(32, activation='relu', name="dense4"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(16, activation='relu', name="dense5"),
        layers.Dense(3, activation="softmax", name="output")
    ])

    ######################### Training ####################
    # we choose the nadam optimizer because it empirically works very well with our data
    model.compile(optimizer='nadam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    # Stop training and restore the best version of the model ever
    # when we spend too many epochs without improving (see the patience parameter)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                               patience=EARLY_STOPPING_PATIENCE, 
                                               verbose=1, 
                                               mode="max", 
                                               restore_best_weights=True)
    
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=early_stop, 
                        verbose=2)
    
    # save the trained model for later use
    model.save("trained_agent")


    ###################### Plotting Training Results ################################
    if PLOT_AFTER_TRAINING:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
