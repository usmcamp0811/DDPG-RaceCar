import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json
import gym
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from collections import deque

OU = OU()  # Ornstein-Uhlenbeck Process



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
def playGame(train_indicator=0, render=True):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 3000
    GAMMA = 0.7
    TAU = 0.0001  # Target Network HyperParameters
    LRA = 0.001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    EXPLORE = 1000000000.
    episode_count = 2000
    max_steps = 100

    step = 0
    epsilon = 1


    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 96*96*3  # of sensors input
    # Generate a CarRacing environment
    env = gym.make('CarRacing-v0')

    observation = env.reset()

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        current_screen_state = observation.flatten()  #observation is given to us from py gym enviroment.
        total_reward = 0
        for j in range(max_steps):
            if render == True:
                env.render()
            loss = 0
            epsilon -= 1.0 / EXPLORE
            action = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim]) #not positive I know what this is but think it has to do w/ OU.py

            a_t_original = actor.model.predict(current_screen_state.reshape(1, current_screen_state.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], -0.2, 0.8, 0.30)  #I know this has to do with the OU stuff
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.6, .7, 0.10)   # but thats all I know
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)
            # The following code do the stochastic brake
            if random.random() <= 0.1:
               print("********Now we apply the brake***********")
               noise_t[0][0] = train_indicator * max(epsilon, 0)
               noise_t[0][1] = train_indicator * max(epsilon, 0)
               noise_t[0][2] = train_indicator * max(epsilon, 0)
            action[0][0] = a_t_original[0][0] + noise_t[0][0]
            action[0][1] = a_t_original[0][1] + noise_t[0][1]
            action[0][2] = a_t_original[0][2] + noise_t[0][2]
            # if random.random() <= 0.1:
            #
            #     print("********Go Straight and put on the Gas***********")
            #     action[0] = [0,0.07,0]

            observation, r_t, done, info = env.step(action[0])
            current_screen_state_1 = observation.flatten()

            buff.add(current_screen_state, action[0], r_t, current_screen_state_1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch]) #guessing its if it was done or not
            y_t = np.asarray([e[1] for e in batch]) #got me what y_t.. looks like its historic actions

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            current_screen_state = current_screen_state_1

            print("Episode", i, "Step", step, "Action", action, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if train_indicator:
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

            print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
            print("Total Step: " + str(step))
            print("")
    env.reset()
     # This is for shutting down TORCS
    print("Finish.")

playGame(train_indicator=1, render=True)


