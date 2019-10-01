from ssbm_gym import SSBMEnv
from reward import isDying
from embed import embedGame
from copy import deepcopy


class ssbm_wrapper():
    def __init__(self, env, frame_limit = 100000, pid=0):
        self.env = env
        self.frame_limit = frame_limit
        self.pid = pid
        self.obs = None
        self.prev_obs = None

        self.action_space = self.env.action_space


    def computeReward(self):
        r = 0.0
        if self.prev_obs is not None:
            if not isDying(self.prev_obs.players[self.pid]) and isDying(self.obs.players[self.pid]):
                r -= 1.0
        
        # Custom reward
        r += self.obs.players[self.pid].y / 100.0 / 60.0

        return r

    def is_terminal(self):
        return self.obs.frame >= self.frame_limit

    def reset(self):
        return embedGame(self.env.reset())
    
    def step(self, action):
        if self.obs is not None:
            self.prev_obs = deepcopy(self.obs)
        
        obs = self.env.step([action])
        self.obs = obs
        reward = self.computeReward()
        done = self.is_terminal()

        infos = dict({'frame': self.obs.frame})

        return embedGame(self.obs), reward, done, infos
