import torch.nn as nn
from config import Args

class Net(nn.Module):

    def __init__(self, args:Args):
        super(Net, self).__init__()
        self.neuralnetwork = nn.Sequential( 
            nn.Linear(args.valueStackSize*args.numberOfLasers, 4, bias= True), #stacking previous distances along with action
            nn.ReLU(),  
            nn.Linear(4, 3, bias= True), #stacking previous distances along with action
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(dim=1)
        ) 
        self.apply(self.initializeWeights)

    def initializeWeights(self, net):
        if isinstance(net, nn.Linear):
            nn.init.uniform_(net.weight)
            nn.init.uniform_(net.bias)

    def forward(self, input):
        output = self.neuralnetwork(input)
        return output