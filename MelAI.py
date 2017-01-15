import tensorflow as tf
from MelAPI.cpu import CPU
import MelAPI.ssbm as ssbm
from model import Network
import numpy as np
from keras import backend as K


class Agent(CPU):
    def __init__(self, character='falcon'):
        super().__init__(character)
        self.net = Network()

    def play(self):
            pad = self.pads[0]
            state = self.state.players[0]
            # print(state)
            policy = self.net.act([state])
            # if self.state.frame % 60 == 0:
            #     print(policy)

            action = np.argmax(policy)
            controller = ssbm.simpleControllerStates[action]
            pad.send_controller(controller.realController())

agent = Agent()
with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    agent.run()
