import gym
import os
from model import TD3
from replaybuffer import ReplayBuffer
os.add_dll_directory("C://Users//nrmlg//.mujoco//mjpro150//bin")
import mujoco_py
import datetime as dt
import tensorflow as tf
import datetime as dt


# initialise the environment
env = gym.make("Ant-v3")
# env = wrappers.Monitor(env, save_dir, force = True) 
env.seed(0)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])

current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
policy = TD3(state_dim, action_dim, max_action, current_time=current_time, summaries=True)
policy.actor.load_weights('./models/{}/actor_{}'.format("20220425-133157", "6"))

state = env.reset()
while True:
    action = policy.select_action(state)
    env.render()
    next_state, reward, done, info = env.step(action)
    state = next_state