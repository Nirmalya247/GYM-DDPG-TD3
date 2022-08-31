import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import time
import random

env = gym.make("MountainCarContinuous-v0")
stateSpace = env.observation_space.shape[0] # 3
actionSpace = env.action_space.shape[0] # 1
maxAction = env.action_space.high[0]
minAction = env.action_space.low[0]

print("Size of State Space ->  {}".format(stateSpace))
print("Size of Action Space ->  {}".format(actionSpace))

print("Max Value of Action ->  {}".format(maxAction)) # 2
print("Min Value of Action ->  {}".format(minAction)) # -2

def createActor():
    # state
    lastInit = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    # input layer
    inputLayer = layers.Input(shape=(stateSpace,))
    # hidden layer
    hiddenLayer = layers.Dense(8, activation="relu")(inputLayer)
    hiddenLayer = layers.Dense(8, activation="relu")(hiddenLayer)
    hiddenLayer = layers.Dense(4, activation="relu")(hiddenLayer)
    # output layer
    outputLayer = layers.Dense(actionSpace, activation="tanh", kernel_initializer=lastInit)(hiddenLayer)
    outputLayer = outputLayer * maxAction
    model = tf.keras.Model(inputLayer, outputLayer)
    return model

def createCritic():
    # state
    stateInputLayer = layers.Input(shape=(stateSpace))
    stateHiddenLayer = layers.Dense(8, activation="relu")(stateInputLayer)
    stateHiddenLayer = layers.Dense(8, activation="relu")(stateHiddenLayer)
    # action
    actionInputLayer = layers.Input(shape=(actionSpace))
    actionHiddenLayer = layers.Dense(8, activation="relu")(actionInputLayer)
    actionHiddenLayer = layers.Dense(8, activation="relu")(actionHiddenLayer)
    # hidden layer
    concat = layers.Concatenate()([stateHiddenLayer, actionHiddenLayer])

    hiddenLayer = layers.Dense(16, activation="relu")(concat)
    hiddenLayer = layers.Dense(16, activation="relu")(hiddenLayer)
    outputLayer = layers.Dense(1)(hiddenLayer)
    # output layer
    model = tf.keras.Model([stateInputLayer, actionInputLayer], outputLayer)
    return model

def policy(state):
    action = tf.squeeze(actor(state))
    legalAction = action.numpy() + np.random.normal(0, (1 / 4) * std_dev, actionSpace)
    legalAction = np.clip(legalAction, minAction, maxAction)
    return [np.squeeze(legalAction)]

actor = createActor()
actor.load_weights('model/actor/actor')
critic = createCritic()
critic.load_weights('model/critic/critic')


std_dev = 0.2

totalEpisodes = 10000
memorySize = 20000
batchSize = 128
gamma = 0.99
tau = 0.005
renderModel = True



def main():
    totalStep = 0
    for episode in range(totalEpisodes):
        episodeReward = 0
        episodeStep = 0
        state = env.reset()
        done = False
        swingMode = 0
        lastRightLeft = 0
        while not done:
            episodeStep = episodeStep + 1
            totalStep = totalStep + 1
            if renderModel:
                env.render()
            state2D = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = policy(state2D)
            
            nextState, reward, done, info = env.step(action)

            state = nextState
            episodeReward += reward

            if done:
                print('episodeReward: {} episodeStep: {} totalStep: {} episode: {}'.format(episodeReward, episodeStep, totalStep, episode))
                break
    env.close()

main()