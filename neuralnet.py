import torch.nn as nn
from config import Args

class Net(nn.Module):

    def __init__(self, args:Args):
        super(Net, self).__init__()
        self.neuralnetwork = nn.Sequential( 
            nn.Linear(args.valueStackSize*(args.numberOfLasers + 1), int(args.valueStackSize*(args.numberOfLasers + 1)/2), bias = True), #stacking previous distances along with action, with add
            nn.ReLU(),  
            nn.Linear(int(args.valueStackSize*(args.numberOfLasers + 1)/2), 3, bias = True), #stacking previous distances along with action
            nn.ReLU(),
            nn.Linear(3, 3, bias = True),
            nn.ReLU(),
        ) 

    def forward(self, input):
        output = self.neuralnetwork(input)
        return output