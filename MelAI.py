import tensorflow as tf
import ctype_tf as ct
from MelAPI.cpu import CPU
import MelAPI.util as util
# from MelAPI.pad import Button
import MelAPI.ssbm as ssbm
import state_embedding as embed
import numpy as np


class Network():
    def __init__(self):
        self.embedPlayer = embed.PlayerEmbedding(action_space=64)
        self.action_size = len(ssbm.simpleControllerStates)
        self.inputs = ct.inputCType(ssbm.PlayerMemory, [1], "bot")
        x = self.embedPlayer(self.inputs)
        x = tf.contrib.layers.fully_connected(x, self.action_size)
        self.policy = tf.nn.softmax(x)

    def act(self, state):
        sess = tf.get_default_session()
        feed_dict = dict(util.deepValues(util.deepZip(self.inputs, ct.vectorizeCTypes(ssbm.PlayerMemory, state))))
        return sess.run(self.policy, feed_dict)


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
    sess.run(tf.global_variables_initializer())
    agent.run()
