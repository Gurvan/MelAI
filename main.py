from ssbm_gym import SSBMEnv
from env_wrapper import ssbm_wrapper
import time
import argparse

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

env = ssbm_wrapper(SSBMEnv(**args.__dict__), frame_limit = 3000)


import torch
from embed import embedGame
from model import Actor, gamesTensors

print(env.action_space)
actor = Actor(env.action_space.n)

obs = env.reset()
action = env.action_space.from_index(actor(obs))

t = time.time()
while True:
    obs, reward, done, infos = env.step(action)
    print("Frame: ", infos['frame'], "\tReward: ", round(reward, 3))
    if done:
        break
    with torch.no_grad():
        action = env.action_space.from_index(actor(obs))

print(time.time() - t)