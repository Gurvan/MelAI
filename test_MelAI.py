from MelAPI.cpu import CPU
#from MelAPI.pad import Button
import MelAPI.ssbm as ssbm
import random


class myCPU(CPU):
    def __init__(self, character = 'falcon'):
        super().__init__(character)   
        self.tictoc = 0
    
    """    
    def play(self):
        action = random.randint(0,53)
        pad = self.pads[0]
        controller = ssbm.simpleControllerStates[action]
        pad.send_controller(controller.realController())
        #self.spam(Button.Y)
    
    def play(self):
        pad = self.pads[0]
        x0 = self.state.players[0].x
        x1 = self.state.players[1].x
        controller = ssbm.RealControllerState()
        if x0 > x1:
            stick = ssbm.Stick(1, 0.5)
        else:
            stick = ssbm.Stick(0, 0.5)
        controller.stick_MAIN = stick
        pad.send_controller(controller)
    """
    def play(self):
        pad = self.pads[0]
        hitlag = self.state.players[1].hitlag_frames_left
        controller = ssbm.RealControllerState()      
        if hitlag > 0:
            stick = ssbm.Stick(self.tictoc, 0)
            self.tictoc = 1-self.tictoc
        else:
            stick = ssbm.Stick(0.5, 0.5)
        controller.stick_MAIN = stick
        pad.send_controller(controller)
                
cpu = myCPU('fox')
cpu.run()