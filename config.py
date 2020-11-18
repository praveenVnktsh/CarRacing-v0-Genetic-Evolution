import torch
import os
import numpy as np


class Args():

    def __init__(self):
        self.checkpoint = 0
        trial = 0
        self.test = False

        self.saveInterval = 10


        # evolution parameters
        self.nAgents = 100
        self.nSurvivors = 20
        self.mutationPower = 0.02
        self.nAvg = 1

        # environment parameters
        self.action_repeat = 4
        self.seed = 0
        self.numberOfLasers = 5
        self.deathThreshold = 2000
        self.deathByGreeneryThreshold = 35
        self.maxDistance = 100
        

        #model parameters
        self.valueStackSize = 8

        #agent parameters
        self.actionMultiplier = np.array([2., 1.0, 1.0])
        self.actionBias = np.array([-1.0, 0.0, 0.0])



        #logistical parameters
        saveloc = 'model/train_' + str(trial) + '/'
        self.saveLocation = saveloc

        os.makedirs(self.saveLocation, exist_ok = True)
        f = open(saveloc + 'params.json','w')
        f.write(str(self.getParamsDict()))
        f.close()

    def getParamsDict(self):
        ret = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        print('\nHYPERPARAMETERS = ', ret)
        return ret
    
    def actionTransformation(self, action):
        return action*self.actionMultiplier + self.actionBias
        

def configure():
    
    args = Args()
    
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")
    torch.manual_seed(args.seed)
    if useCuda:
        torch.cuda.manual_seed(args.seed)
    return args, useCuda, device