import tensorflow as tf
import ctype_tf as ct
import MelAPI.util as util
import MelAPI.ssbm as ssbm
import state_embedding as embed
from keras.layers import Dense


class Network():
    def __init__(self):
        self.embedPlayer = embed.PlayerEmbedding(action_space=64)
        self.action_size = len(ssbm.simpleControllerStates)
        self.inputs = ct.inputCType(ssbm.PlayerMemory, [1], "bot")
        x = self.embedPlayer(self.inputs)
        # x = tf.contrib.layers.fully_connected(x, self.action_size)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        self.logits = Dense(self.action_size, activation='linear')(x)
        self.value = Dense(1, activation='linear')(x)
        self.policy = tf.nn.softmax(self.logits)

    def act(self, state):
        sess = tf.get_default_session()
        feed_dict = dict(util.deepValues(util.deepZip(self.inputs, ct.vectorizeCTypes(ssbm.PlayerMemory, state))))
        return sess.run([self.policy, self.value], feed_dict)
