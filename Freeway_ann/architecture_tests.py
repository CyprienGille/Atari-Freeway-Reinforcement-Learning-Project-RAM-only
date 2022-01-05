"""
Script used to make tests on the performance of an agent depending on the depth
of its neural network.

Note: due to the fact that we need to run a full game for each model
to get its final score, this script can take some time to complete
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gym

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

if __name__ == "__main__":

    ######################### Constants/params ####################
    TRAIN_PROP = 0.85 # The rest will be reserved for validation
    BATCH_SIZE = 32 # not very important in our case
    EPOCHS = 300 # max number of epochs, almost never reached

    # how many epochs without improvement to wait before stopping training
    EARLY_STOPPING_PATIENCE = 30
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

    #########################################################################
    ################################## Architecture Testing #################
    #########################################################################
    class ModelInfos:
        """Small dataclass to store models information easily"""
        def __init__(self):
            self.train_accuracies = []
            self.val_accuracies = []
            self.n_epochs_needed = []
            self.depths = []
        
        def store_model_info(self, history, n_layers):
            self.train_accuracies.append(np.max(history.history["accuracy"]))
            self.val_accuracies.append(np.max(history.history["val_accuracy"][-1]))
            self.n_epochs_needed.append(len(history.history["accuracy"]))
            self.depths.append(n_layers)

    
    def get_model(depth, width=48):
        layer_list = [layers.Dense(128, input_shape=(128,), activation='relu', name="input")]
        layer_list += [layers.Dense(width, activation='relu', name=f"dense{i}") for i in range(depth)]
        layer_list += [layers.Dense(3, activation='softmax', name="output")]
        return Sequential(layer_list)

    def run_simul(model):
        """make the model play a game to see what score it gets"""
        move_ids = [0, 1, 2]
        env = gym.make('Freeway-ram-v0')
        env.reset()
        score = 0

        past, _, _, _ = env.step(0)
        future, rew, done, _ = env.step(0)
        while not done:
            score += rew
            data = np.array(future) - np.array(past)
            data = tf.convert_to_tensor(data)
            data = tf.expand_dims(data, 0) # add batch dimension

            pred = model.predict(data)[0]
            action = move_ids[np.argmax(pred)]

            past = future
            future, rew, done, _ = env.step(action)

        env.close()
        return score

    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                patience=EARLY_STOPPING_PATIENCE, 
                                                verbose=1, 
                                                mode="max",
                                                restore_best_weights=True)
    infos = ModelInfos()
    scores = []
    for depth in range(15):
        model = get_model(depth)

        model.compile(optimizer='nadam', 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])


        print(f"\nInitiating training for ANN with {depth} hidden layer(s)...")
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=EPOCHS,
                            callbacks=early_stop, 
                            verbose=0)
        
        infos.store_model_info(history, depth)
        scores.append(run_simul(model))

    
    plt.figure(figsize=(13, 8))
    plt.subplot(131)
    plt.plot(infos.depths, infos.n_epochs_needed, "o-", label="Epochs")
    plt.legend()
    plt.title("Epochs needed before convergence per depth")

    plt.subplot(132)
    plt.plot(infos.depths, infos.train_accuracies, "o-", label="Training Accuracies")
    plt.plot(infos.depths, infos.val_accuracies, "o-", label="Validation Accuracies")
    plt.legend()
    plt.title("Accuracies after training per depth")

    plt.subplot(133)
    plt.plot(infos.depths, scores, "o-", label="Score")
    plt.legend()
    plt.title("Score reached per depth")
    
    plt.show() # use the UI to save the picture if it is interesting
