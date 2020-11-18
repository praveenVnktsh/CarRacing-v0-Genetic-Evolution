import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.neuralnetwork = nn.Sequential( 
            nn.Linear(40, 128, bias= True), #stacking previous distances along with action
            nn.ReLU(),  
            nn.Linear(128, 3),
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

model = MyModel()
print(model.state_dict())