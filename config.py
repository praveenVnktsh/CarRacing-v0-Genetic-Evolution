import torch
import os
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Args():

    def __init__(self):
        # Logistical Parameters
        self.checkpoint = 0
        trial = 14
        self.test = False
        
        # evolution parameters
        
        self.mutationPower = 0.05
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
        self.font = 'Calibri'
        self.carImagePath = "environment/data/car.png"
        self.trackPath = "environment/data/track.png"
        self.startingPositionX = 175
        self.startingPositionY = 435
        self.cameraOffset = 50
        self.cameraHeight = 500
        self.width = 1500
        self.height = 1500
        self.bgColor = (120, 120, 120)

        #Car properties
        self.anglesToSee = [-50, -40 , -30, -20, -10, 0, 10, 20, 30, 40, 50]
        self.numberOfLasers = len(self.anglesToSee)
        self.maxSteering = 20
        self.maxAcceleration = 0.4
        self.maxVelocity = 20
        self.maxBraking = 0.4
        self.freeDeceleration = 0.1
        self.angle = 85
        self.distanceToSee = 150
        self.steeringmaxcut = 5

        # Settings
        self.nSurvivors = 20
        self.numberOfCars = 100
        if self.test:
            self.numberOfCars = 5
            self.nSurvivors = self.numberOfCars
            
        
        self.render = True
        self.deathThreshold = 4500



        #logistical parameters
        saveloc = 'model/train_' + str(trial) + '/'
        self.saveLocation = saveloc

        os.makedirs(self.saveLocation, exist_ok = True)
        f = open(saveloc + 'params.json','w')
        json.dump(self.getParamsDict(), f, indent=4, sort_keys=True, cls=NumpyEncoder)
        f.close()

    def getParamsDict(self):
        ret = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        return ret
    
    def actionTransformation(self, action):
        return action*self.actionMultiplier + self.actionBias
        

def configure():
    
    args = Args()
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")
    return args, useCuda, device