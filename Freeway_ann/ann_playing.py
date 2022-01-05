"""
Script to make the ann play a game of Freeway. 
This script is supposed to be ran after ann_training.py.

Note: If you want to interrupt the game mid-way, kill the terminal you ran this from.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model (saved in ann_training.py)
model = keras.models.load_model('trained_agent')

# The moves our agent can make (stay, up, down)
move_ids = [0, 1, 2]


    
env = gym.make('Freeway-ram-v0')
env.reset()
env.render()

past, _, _, _ = env.step(env.action_space.sample())
future, _, done, _ = env.step(env.action_space.sample())
while not done:
    data = np.array(future) - np.array(past)
    data = tf.convert_to_tensor(data)
    data = tf.expand_dims(data, 0) # add the batch dimension
    
    pred = model.predict(data)[0]
    action = move_ids[np.argmax(pred)]

    past = future
    future, _, done, _ = env.step(action)
    env.render()

env.close()
