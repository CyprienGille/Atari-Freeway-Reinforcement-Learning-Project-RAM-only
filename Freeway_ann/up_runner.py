"""
Script to run a game with a player that always runs up, or can stay still with a custom probability
"""


import gym

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng= default_rng()
moves = np.array([0, 1])

# Run a game with an agent that stays still/runs up with first/second probability
def run_simul(probas=np.array([0.0, 1.0])):
    env = gym.make('Freeway-ram-v0')
    env.reset()
    score = 0
    _, rew, done, _ = env.step(1) # go up on first frame
    while not done:
        score += rew
        _, rew, done, _ = env.step(rng.choice(moves, size=1, replace=True, p=probas))

    env.close()
    return score

X = np.linspace(0.0, 1.0, 60)
Y = []
for p0 in X:
    p1 = 1.0 - p0
    p = np.array([p0, p1])
    Y.append(run_simul(p))


plt.plot(X, Y, "r", label="Score")
plt.legend()
plt.title("Score per probability of staying still instead of moving up")
plt.show()
    
