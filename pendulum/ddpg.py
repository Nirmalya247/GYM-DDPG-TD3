import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import time
import random

env = gym.make("Pendulum-v1")
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
    hiddenLayer = layers.Dense(256, activation="relu")(inputLayer)
    hiddenLayer = layers.Dense(256, activation="relu")(hiddenLayer)
    # output layer
    outputLayer = layers.Dense(actionSpace, activation="tanh", kernel_initializer=lastInit)(hiddenLayer)
    outputLayer = outputLayer * maxAction
    model = tf.keras.Model(inputLayer, outputLayer)
    return model

def createCritic():
    # state
    stateInputLayer = layers.Input(shape=(stateSpace))
    stateHiddenLayer = layers.Dense(16, activation="relu")(stateInputLayer)
    stateHiddenLayer = layers.Dense(32, activation="relu")(stateHiddenLayer)
    # action
    actionInputLayer = layers.Input(shape=(actionSpace))
    actionHiddenLayer = layers.Dense(32, activation="relu")(actionInputLayer)
    # hidden layer
    concat = layers.Concatenate()([stateHiddenLayer, actionHiddenLayer])

    hiddenLayer = layers.Dense(256, activation="relu")(concat)
    hiddenLayer = layers.Dense(256, activation="relu")(hiddenLayer)
    outputLayer = layers.Dense(1)(hiddenLayer)
    # output layer
    model = tf.keras.Model([stateInputLayer, actionInputLayer], outputLayer)
    return model

def policy(state):
    action = tf.squeeze(actor(state))
    legalAction = action.numpy() + np.random.normal(0, (1 / 4) * std_dev, actionSpace)
    legalAction = np.clip(legalAction, minAction, maxAction)
    return [np.squeeze(legalAction)]

@tf.function
def updateTarget(targetWeights, weights, tau):
    for (a, b) in zip(targetWeights, weights):
        a.assign(b * tau + a * (1 - tau))

actor = createActor()
# actor.load_weights('pendulum/actor/actor')
actorTarget = createActor()
actorTarget.set_weights(actor.get_weights())
critic = createCritic()
# critic.load_weights('pendulum/critic/critic')
criticTarget = createCritic()
criticTarget.set_weights(critic.get_weights())


std_dev = 0.2
critic_lr = 0.002
actor_lr = 0.001

criticOptimizer = tf.keras.optimizers.Adam(critic_lr)
actorOptimizer = tf.keras.optimizers.Adam(actor_lr)

totalEpisodes = 100
memorySize = 100000
batchSize = 64
gamma = 0.99
tau = 0.005
renderModel = True



class Trainer:
    def __init__(self, mamorySize=100000, batchSize=64):
        self.memory = deque(maxlen=mamorySize)
        self.batchSize = batchSize
        self.tempMemory = [ ]
        self.t0 = time.time()
    def addTemp(self, sate, action, nextSate, done):
        self.tempMemory.append([sate, action, nextSate, done])
    def addReward(self, reward):
        for x in self.tempMemory:
            self.memory.append([x[0], x[1], [reward], x[2], x[3]])
        self.tempMemory = [ ]
    
    @tf.function
    def update(self, states, actions, rewards, nextStates):
        with tf.GradientTape() as tape:
            nextActions = actorTarget(nextStates, training=True)
            y = rewards + gamma * criticTarget([nextStates, nextActions], training=True)
            cricics = critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - cricics))

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        criticOptimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor(states, training=True)
            cricics = critic([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(cricics)

        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actorOptimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))

    def train(self):
        batchSize = min(self.batchSize, len(self.memory))
        batch = random.sample(self.memory, batchSize)

        states = tf.convert_to_tensor([t[0] for t in batch])
        actions = tf.convert_to_tensor([t[1] for t in batch])
        rewards = tf.convert_to_tensor([t[2] for t in batch])
        rewards = tf.cast(rewards, dtype=tf.float32)
        nextStates = tf.convert_to_tensor([t[3] for t in batch])

        self.update(states, actions, rewards, nextStates)
    
    def saveModels(self):
        actorTarget.save_weights('model/actor/actor')
        criticTarget.save_weights('model/critic/critic')
    def saveData(self, episodeReward, episodeStep, totalStep, episodeNo):
        file1 = open("ddpg1.txt", "a")
        file1.write('{}, {}, {}, {}'.format(episodeReward, episodeStep, totalStep, episodeNo, self.t0 - time.time()))
        file1.write("\n")
        file1.close()

def main():
    trainer = Trainer(memorySize, batchSize)
    totalStep = 0
    for episode in range(totalEpisodes):
        episodeReward = 0
        episodeStep = 0
        state = env.reset()
        done = False
        while not done:
            episodeStep = episodeStep + 1
            totalStep = totalStep + 1
            if renderModel:
                env.render()
            state2D = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = policy(state2D)
            
            nextState, reward, done, info = env.step(action)
            print(reward, critic([tf.convert_to_tensor([state]), tf.convert_to_tensor([action])]))
            trainer.addTemp(state, action, nextState, done)
            trainer.addReward(reward)

            if len(trainer.memory) > batchSize:
                trainer.train()
                updateTarget(actorTarget.variables, actor.variables, tau)
                updateTarget(criticTarget.variables, critic.variables, tau)

            state = nextState
            episodeReward += reward

            if done:
                trainer.saveModels()
                trainer.saveData(episodeReward, episodeStep, totalStep, episode)
                print('episodeReward: {} episodeStep: {} totalStep: {} episode: {}'.format(episodeReward, episodeStep, totalStep, episode))
                break
    env.close()

main()