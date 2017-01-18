import tensorflow as tf
from agent import Agent
import numpy as np
import ctype_tf as ct
import MelAPI.util as util
import MelAPI.ssbm as ssbm

from collections import namedtuple
import time
from scipy.signal import lfilter
from keras import backend as K

Batch = namedtuple("Batch", ["si", "a", "adv", "r"])


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    batch_si = rollout.states
    batch_a = np.array(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    return Batch(batch_si, batch_a, batch_adv, batch_r)


class Trainer():
    def __init__(self):
        self.agent = Agent('fox')
        net = self.agent.net
        self.global_step = tf.get_variable("global_step",
                                           [],
                                           tf.int32,
                                           initializer=tf.constant_initializer(0, dtype=tf.int32),
                                           trainable=False)

        self.ac = tf.placeholder(tf.float32, [None, net.action_size], name='ac')
        self.adv = tf.placeholder(tf.float32, [None], name='adv')
        self.r = tf.placeholder(tf.float32, [None], name='r')

        log_prob = tf.log(net.policy)
        prob = net.policy
        policy_loss = - tf.reduce_sum(tf.reduce_sum(log_prob * self.ac, [1]) * self.adv)
        value_loss = tf.nn.l2_loss(net.value - self.r)
        entropy = - tf.reduce_sum(prob * log_prob)

        batch_size = tf.shape(self.ac)[0]
        self.loss = policy_loss + 0.5 * value_loss - entropy * 0.01
        inc_step = self.global_step.assign_add(batch_size)

        optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.train_op = tf.group(optimizer, inc_step)

    def start(self, sess):
        self.agent._start(sess)

    def pull_batch_from_queue(self):
        return self.agent.queue.get()

    def process(self, sess):
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        if len(batch.a) > 120:
            print("Learning...")
            feed_dict = dict(util.deepValues(util.deepZip(self.agent.net.inputs, ct.vectorizeCTypes(ssbm.GameMemory, batch.si))))
            feed_dict.update({self.ac: batch.a,
                              self.adv: batch.adv,
                              self.r: batch.r})

            fetched = sess.run([self.train_op, self.global_step], feed_dict)


num_global_steps = 1000000000
trainer = Trainer()

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

now = time.time()
sv = tf.train.Supervisor(logdir='log',
                         saver=saver,
                         init_op=init_op,
                         summary_op=None,
                         ready_op=tf.report_uninitialized_variables(),
                         global_step=trainer.global_step,
                         save_model_secs=30)
print("Supervisor generation:", time.time()-now)

with sv.managed_session() as sess, sess.as_default():
    print("Begin training.")
    K.set_session(sess)
    global_step = sess.run(trainer.global_step)
    trainer.start(sess)
    while global_step < num_global_steps:
        trainer.process(sess)
        global_step = sess.run(trainer.global_step)

sv.stop()
