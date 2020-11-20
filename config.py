import torch
import os
import numpy as np


class Args():

    def __init__(self):
        self.checkpoint = 0
        trial = 4
        self.test = False
        

        self.saveInterval = 10


        # evolution parameters
        
        self.mutationPower = 0.03
        self.nAvg = 1

        # environment parameters
        self.action_repeat = 4
        self.numberOfLasers = 5
        self.deathThreshold = 2000
        self.deathByGreeneryThreshold = 35
        self.maxDistance = 100
        

        #model parameters
        self.valueStackSize = 1

        #agent parameters
        self.actionMultiplier = np.array([2., 1.0, 1.0])
        self.actionBias = np.array([-1.0, 0.0, 0.0])


        #Environment properties        
        self.carImagePath = "environment/data/car.png"
        self.trackPath = "environment/data/track.png"
        self.startingPositionX = 118
        self.startingPositionY = 435
        self.cameraHeight = 300
        self.width = 1500
        self.height = 1500
        self.bgColor = (120, 120, 120)

        #Car properties
        self.anglesToSee = [-50, -25, 0, 25, 50]
        self.numberOfLasers = len(self.anglesToSee)
        self.maxSteering = 90
        self.maxAcceleration = 0.1
        self.maxVelocity = 1.5
        self.freeDeceleration = 0.03
        self.distanceToSee = 150
        self.maxBraking = 0.05
        # Settings
        self.nSurvivors = 20
        self.numberOfCars = 100
        self.render = True



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
    # torch.manual_seed(args.seed)
    # if useCuda:
    #     torch.cuda.manual_seed(args.seed)
    return args, useCuda, device