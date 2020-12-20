from comet_ml import Experiment
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Car, Environment
from config import Args, configure
from API_KEYS import api_key, project_name
import torch
import os
import glob
import time
configs, use_cuda,  device = configure()

## SET LOGGING
experiment = Experiment(project_name = project_name,  api_key = api_key)
experiment.log_parameters(configs.getParamsDict())
    

def getTrainTest( isTest = False, experiment = None,):
    if isTest:
        return experiment.test()
    return experiment.train()


def mutateWeightsAndBiases(agents, configs:Args):
    nextAgents = []

    if configs.test == True:
        for i in range(configs.numberOfCars):
            pair = agents[i]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            nextAgents.append(agentNet)
    else:
        for i in range(configs.numberOfCars):
            pair = agents[i % len(agents)]
            agentNet = Agent(configs, device, stateDict = pair[0].getParams())
            for param in agentNet.net.parameters():
                param.data += configs.mutationPower * torch.randn_like(param)
            nextAgents.append(agentNet)

    return nextAgents

def saveWeightsAndBiases(agentDicts, generation, configs:Args):
    loc = configs.saveLocation +'generation_'+str(generation) +  '/' 
    os.makedirs(loc, exist_ok = True)
    for i in range(len(agentDicts)):
        torch.save(agentDicts[i], loc + str(i) +  '-AGENT.pkl')



if __name__ == "__main__":
    print('-------------BEGINNING EXPERIMENT--------------')
    
    
    
    currentAgents = []
    if configs.checkpoint != 0:
        for location in sorted(glob.glob(configs.saveLocation +'generation_'+str(configs.checkpoint) +  '/*')):
            print('LOADING FROM',location)
            statedict = torch.load(location)
            currentAgents.append(statedict)
        
        currentAgents = mutateWeightsAndBiases(currentAgents, configs)
        print('-> Loaded agents from checkpoint', configs.checkpoint)
    else:
        for spawnIndex in range(configs.numberOfCars):
            agent = Agent(configs, device)
            currentAgents.append(agent)


    env = Environment(configs)

    with getTrainTest(configs.test, experiment):
        rewards = np.zeros((configs.numberOfCars, ))
        for generationIndex in range(configs.checkpoint, 100000):
            env.reset()
            action = np.zeros((configs.numberOfCars, 3)) 
            state = np.ones((configs.numberOfCars, (configs.numberOfLasers + 1)*configs.valueStackSize))*configs.distanceToSee #+1 is for the velocity component
            dead = np.zeros((configs.numberOfCars, ))
            rewards = np.zeros((configs.numberOfCars, ))
            nextAgents = []

            startTime = time.time()
            for timestep in range(configs.deathThreshold):
                for agentIndex in range(len(currentAgents)):
                    if dead[agentIndex] != 1.0:
                        action[agentIndex] = currentAgents[agentIndex].chooseAction(state[agentIndex])
                action = action.clip(0.0, 1.0)
                action[:, 0] *=2 
                action[:, 0] -=1

                if not configs.test:
                    if (generationIndex) % 5 == 0:
                        render = True
                    else:
                        render = True
                else:
                    render = True
                
                logData = {
                    'Generation':generationIndex, 
                    'Timestep':timestep, 
                    'Alive':configs.numberOfCars - np.sum(dead),
                    'Fitness': np.round(np.mean(rewards), 3),
                    }


                state, dead, rewards = env.step(action, data = logData, render = render)


                if 0.0 not in dead:
                    break

            avgScore = np.mean(rewards)
            experiment.log_metric("fitness", np.mean(avgScore) , step= generationIndex)

            print('Generation', generationIndex,'Complete in ',time.time() - startTime , 'seconds')
            print('FITNESS = ', avgScore)
            print('---------------')
            
            if not configs.test:
                temp = [[currentAgents[agentIndex], rewards[agentIndex]] for agentIndex in range(len(currentAgents)) ]
                currentAgents = sorted(temp, key = lambda ag: ag[1], reverse = True)
                nextAgents = currentAgents[:configs.nSurvivors]

                currentAgents = mutateWeightsAndBiases(nextAgents, configs)
                if (generationIndex + 1) % 5 == 0:
                    saveWeightsAndBiases(nextAgents, generationIndex, configs)
            else:
                env.saveImage()

        
            
            
            

