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
        self.terminal = False

    def add(self, state, action, reward, value, terminal):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal


class Agent(CPU):
    def __init__(self, character='falcon'):
        super().__init__(character)
        self.net = Network()
        self.lengths = 0
        self.rewards = 0
        self.num_local_steps = 0
        self.rollout = PartialRollout()

    def play(self):
            pad = self.pads[0]
            state = self.state.players[0]
            policy, value = self.net.act([state])
            action = np.random.choice(0, len(ssbm.simpleControllerStates), p=policy)
            controller = ssbm.simpleControllerStates[action]
            pad.send_controller(controller.realController())

agent = Agent()
with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    agent.run()
