import tensorflow as tf
import embed
import MelAPI.ctype_util as ct
import MelAPI.ssbm as ssbm
import MelAPI.util = util

from enum import Enum
from actor_critic import ActorCritic

class Mode(Enum):
    TRAIN = 0
    PLAY = 1
    
models = [
    ActorCritic,
]
models = {model.__name__ : model for model in models}

class RLConfig():
    def __init__(self):
        self.act_every = 3
        self.reward_halflife = 2.0
        self.experience_time = 5
        self.fps = 60 // self.act_every
        self.discount = self.discount = 0.5 ** ( 1.0 / (self.fps*self.reward_halflife) )
        self.experience_length = self.experience_time * self.fps


class Model():
    def __init__(self, path = None, mode = Mode.TRAIN, model = "ActorCritic", debug = False):
        self.model = model
        if path = None:
            self.path = model
        else:
            self.path = path
            
        self.memory = 0    
            
        self.rlConfig = RLConfig()
        self.embedGame = embed.GameEmbedding()
            
        print("Creating model: ", self.model)
        modelType = models[self.model]
        
        self.graph = tf.Graph()
        
        device = '/cpu:0'
        print("Using device ", device)
        
        
        with self.graph.as_default(), tf.device(device):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            state_size = self.embedGame.size
            history_size = state_size + embed.action_size
            self.model = modelType(history_size, embed.action_size, self.global_step, self.rlConfig)
            
            if mode = Mode.TRAIN:
                with tf.name_scope('train'):
                    self.experience = ct.inputCType(ssbm.SimpleStateAction, [None, self.rlConfig.experience_length], "experience")
                    # instantaneous rewards for all but the first state
                    self.experience['reward'] = tf.placeholder(tf.float32, [None, None], name='experience/reward')
            
                    # initial state for recurrent networks
                    self.experience['initial'] = util.deepMap(lambda size: tf.placeholder(tf.float32, [None, size], name="experience/initial"), self.model.hidden_size)
            
                    mean_reward = tf.reduce_mean(self.experience['reward'])
                    
                    states = self.embedGame(self.experience['state'])
                    
                    prev_actions = embed.embedAction(self.experience['prev_action'])
                    states = tf.concat(2, [states, prev_actions])
                    
                    train_length = self.rlConfig.experience_length - self.memory
                    
                    history = [tf.slice(states, [0, i, 0], [-1, train_length, -1]) for i in range(self.memory+1)]
                    self.train_states = tf.concat(2, history)
                    
                    actions = embed.embedAction(self.experience['action'])
                    self.train_actions = tf.slice(actions, [0, self.memory, 0], [-1, train_length, -1])
          
                    self.train_rewards = tf.slice(self.experience['reward'], [0, self.memory], [-1, -1])
                    
                    train_args = dict(
                      states=self.train_states,
                      actions=self.train_actions,
                      rewards=self.train_rewards,
                      initial=self.experience['initial']
                    )
                    
                    self.train_op = self.model.train(**train_args)
                    
                    tf.scalar_summary('reward', mean_reward)
                    merged = tf.merge_all_summaries()
          
                    increment = tf.assign_add(self.global_step, 1)
          
                    misc = tf.group(increment)
          
                    self.run_dict = dict(summary=merged, global_step=self.global_step, train=self.train_op, misc=misc)
          
                    print("Creating summary writer at logs/%s." % self.name)
                    self.writer = tf.train.SummaryWriter('logs/' + self.name, self.graph)
           
            else:
                pass
                    
            self.debug = debug
      
            self.variables = tf.all_variables()
      
            self.saver = tf.train.Saver(self.variables)
      
            self.placeholders = {v.name : tf.placeholder(v.dtype, v.get_shape()) for v in self.variables}
            self.unblobber = tf.group(*[tf.assign(v, self.placeholders[v.name]) for v in self.variables])
      
            tf_config = dict(
              allow_soft_placement=True,
              #log_device_placement=True,
            )
      
            self.sess = tf.Session(
              graph=self.graph,
              config=tf.ConfigProto(**tf_config),
            )