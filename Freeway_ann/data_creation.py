"""
The purpose of this script is to create our database.

Notes: 

- Before you start creating data, you need to have created a 'data' subfolder in the working dir
- The whole point of this file is to allow for resuming of the data creation, through the RESUMING parameter. 
Don't forget to switch it if you want to resume your data creation, or else you will overwrite previously recorded frames.
-  This script is not os-proof and might break on your OS due to os-specific path separators ("/" vs. "\\" for example)
"""

import numpy as np
import gym
from gym.utils.play import play

RESUMING = False # change this parameter to True if you want to append new data to previously recorded data

class Saver():
    """Utilitary class to make the whole saving process easier"""
    
    def __init__(self, directory="./data", features_array_name="arrF.npy", labels_array_name="arrL.npy", resuming=False):
        self.DIR = directory
        self.FEATURES_PATH = self.DIR + "/" + features_array_name
        self.LABELS_PATH = self.DIR + "/" + labels_array_name
        if resuming:
            self.features = list(np.load(self.FEATURES_PATH))
            self.labels = list(np.load(self.LABELS_PATH))
        else:
            self.features = []
            self.labels = []
    
    def gym_callback(self, obs_t, obs_tp1, action, rew, done, info):
        self.store(obs_t, obs_tp1, action)

    
    def store(self, past, future, label):
        datapoint = np.array(future) - np.array(past) # difference to include temporal information
        self.features.append(datapoint)
        self.labels.append(label)
    
    def commit_save(self):
        np.save(self.FEATURES_PATH, np.array(self.features))
        np.save(self.LABELS_PATH, np.array(self.labels))
        print(f"Currently saved data: {len(self.labels)} frames.")
        self.features = []
        self.labels = []



if __name__=="__main__":
    s = Saver(resuming=RESUMING)
    env = gym.make('Freeway-ram-v0')
    env.reset()
    # increase the fps if you want to create data more quickly
    play(env, zoom=3, fps=20, callback=s.gym_callback)
    s.commit_save()