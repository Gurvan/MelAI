import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('../ssbm_gym')

from ssbm_env import SSBMEnv
import time
import argparse
import random

parser = argparse.ArgumentParser()
SSBMEnv.update_parser(parser)

args = parser.parse_args()

args.iso = "/home/gurvan/ISOs/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
args.exe = "dolphin-emu-nogui"
args.zmq = 1
args.gfx = 'Null'
args.audio = 'No audio backend'
args.speed = 1
args.speedhack = True
args.player1 = 'ai'
args.player2 = 'cpu'
args.cpu2 = 7
args.stage = 'battlefield'

env = SSBMEnv(**args.__dict__)

from copy import deepcopy

from embed import embedGame
from model import Actor, gamesTensors

print(env.action_space)
actor = Actor(env.action_space.n)

obs_ = []
obs = env.reset()

while True:
    try:
        embedGame(obs)
        break
    except:
        obs = env.step([env.action_space.sample()])

obs_.append(deepcopy(obs))
print(obs.players[0])
state = gamesTensors([embedGame(obs)])

action = env.action_space.from_index(actor(state))
#print(action)
import time
import torch

t = time.time()

for i in range(3000):
    obs = env.step([action])
    obs_.append(deepcopy(obs))
    state = gamesTensors([embedGame(obs)])
    with torch.no_grad():
        action = env.action_space.from_index(actor(state))
    #print(action)

print(time.time()-t)

from reward import computeRewards

r = computeRewards(obs_)

# for i in range(5):
#     s = env.step([env.action_space.sample()])
#     states.append(embedGame(s, True))

from IPython import embed
embed()


# for i in range(1000):
#     # s = env.step([random.choice(env.action_space)])
#     s = env.step([env.action_space.sample()])
#     p0 = embedPlayer(s.players[0])
#     print(s.frame)
#     #print(p0['action_state'])

env.close()

