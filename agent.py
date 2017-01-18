import tensorflow as tf
from MelAPI.cpu import CPU
import MelAPI.ssbm as ssbm
from model import Network
from reward import computeRewards

import numpy as np
import six.moves.queue as queue

import copy

import sys


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


class Agent(CPU):
    def __init__(self, character='falcon'):
        super().__init__(character)
        self.net = Network()
        self.queue = queue.Queue(2)
        self.rollout = PartialRollout()
        self.previous_state = ssbm.GameMemory()
        self.previous_action = self.one_hot(0, len(ssbm.simpleControllerStates))
        self.previous_value = 0
        self.runback = True

    def one_hot(self, a, l):
        x = np.zeros(l)
        x[a] = 1
        return x

    def _start(self, sess):
        self.sess = sess
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def press_AB(self, pad):
        controller = ssbm.RealControllerState()
        controller.button_A = True
        controller.button_B = True
        pad.send_controller(controller)

    def play(self):
            pad = self.pads[0]
            if not self.previous_state.players:
                self.previous_state = copy.deepcopy(self.state)

            policy, value = self.net.act([self.state])
            action = np.random.choice(len(ssbm.simpleControllerStates), p=policy[0])
            reward = computeRewards([self.previous_state, self.state])

            self.rollout.add(self.previous_state, self.previous_action, reward, self.previous_value)

            if 0 in [self.state.players[0].stock, self.state.players[1].stock]:
                if not (0 in [self.previous_state.players[0].stock, self.previous_state.players[1].stock]):
                    if np.random.random() < 0.25:
                        self.runback = False
                if self.runback:
                    self.press_AB(pad)
                self.queue.put(self.rollout)
                self.rollout = PartialRollout()
            else:
                self.runback = False
                controller = ssbm.simpleControllerStates[action]
                pad.send_controller(controller.realController())
            self.previous_state = copy.deepcopy(self.state)
            self.previous_action = self.one_hot(action, len(ssbm.simpleControllerStates))
            # print(self.previous_action)
            self.previous_value = value
