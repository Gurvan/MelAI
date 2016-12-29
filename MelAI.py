from MelAPI.cpu import CPU
#from MelAPI.pad import Button
import MelAPI.ssbm as ssbm
import random


class myCPU(CPU):
    def __init__(self, character = 'falcon'):
        super().__init__(character)   
        
    def play(self):
        action = random.randint(0,53)
        pad = self.pads[0]
        controller = ssbm.simpleControllerStates[action]
        pad.send_controller(controller.realController())
        #self.spam(Button.Y)
        


cpu = myCPU('falco')
cpu.run()