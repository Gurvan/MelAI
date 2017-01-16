import tensorflow as tf
from MelAPI.cpu import CPU
import MelAPI.ssbm as ssbm
from model import Network
from reward import computeRewards

import numpy as np
from keras import backend as K
import six.moves.queue as queue
from collections import namedtuple
from scipy.signal import lfilter

import copy


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal"])


class PartialRollout(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0

    def add(self, state, action, reward, value):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]


class State():
    def __init__(self):
        self.players = []
        self.frame = 0
        self.menu = 0
        self.stage = 0

    def copy(self, state):
        self.players = state.players
        self.frame = state.frame
        self.menu = state.menu
        self.stage = state.stage


class Agent(CPU):
    def __init__(self, character='falcon'):
        super().__init__(character)
        self.net = Network()
        self.rollout = PartialRollout()
        self.previous_state = ssbm.GameMemory()

    def play(self):
            pad = self.pads[0]
            if not self.previous_state.players:
                self.previous_state = copy.deepcopy(self.state)

            #print(self.state.players[0].x - self.previous_state.players[0].x)
            # print(self.state)
            policy, value = self.net.act([self.state])
            action = np.random.choice(len(ssbm.simpleControllerStates), p=policy[0])
            reward = computeRewards([self.previous_state, self.state])
            if not reward[0] == 0:
                print(reward)

            self.rollout.add(self.previous_state, action, reward, value)

            controller = ssbm.simpleControllerStates[action]
            pad.send_controller(controller.realController())
            self.previous_state = copy.deepcopy(self.state)


agent = Agent()
with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    agent.run()
