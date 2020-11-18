from comet_ml import Experiment
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import Args, configure
from API_KEYS import api_key, project_name
import torch
import os

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
    for i in range(configs.nAgents):
        pair = agents[i % configs.nSurvivors]
        agentNet = Agent(configs, device, stateDict = pair[0]).net
        for param in agentNet.parameters():
            param.data += configs.mutationPower * torch.randn_like(param)
        nextAgents.append(agentNet.state_dict())
    return nextAgents

def saveWeightsAndBiases(agentDicts, generation, configs:Args):
    loc = configs.saveLocation +'generation_'+str(generation) +  '/' 
    os.makedirs(loc, exist_ok = True)
    for i in range(len(agentDicts)):
        torch.save(agentDicts[i], loc + str(i) +  '-AGENT.pkl')



if __name__ == "__main__":
    
    
    env = Env(configs)
    
    currentAgents = []
    if configs.checkpoint != 0:
        for spawnIndex in range(configs.nSurvivors):
            statedict = torch.load(configs.saveLocation +'generation_'+str(configs.checkpoint) +  '/'  + str(spawnIndex) +  '-AGENT.pkl')
            currentAgents.append(statedict)
        currentAgents = mutateWeightsAndBiases(currentAgents, configs)
        print('Loaded agents from checkpoints', configs.checkpoint)
    else:
        for spawnIndex in range(configs.nAgents):
            agent = Agent(configs, device)
            currentAgents.append(agent.getParams())

    nextAgents = []

    with getTrainTest(configs.test, experiment):


        print('-------------BEGINNING EXPERIMENT--------------')
        for generationIndex in range(configs.checkpoint, 100000):
            

            for spawnIndex in range(configs.nAgents):
                agent = Agent(configs, device, stateDict = currentAgents[spawnIndex])
                scores = []

                for episode in range(configs.nAvg): #repeat for 3 iterations
                    score = 0
                    prevState = env.reset()
                    for timestep in range(10000):
                        action = agent.chooseAction(prevState)
                        
                        curState, reward, dead, reasonForDeath = env.step(action, timestep)
                        
                        score += reward
                        prevState = curState

                        if dead:
                            break
                    
                    scores.append(score)
                
                avgScore = np.mean(np.array(scores))
                print('Generation = ', generationIndex, 'Spawn =', spawnIndex, 'Average score =',  avgScore)
                nextAgents.append((agent.getParams(), avgScore))


            currentAgents = sorted(nextAgents, key = lambda ag: ag[1], reverse = True)
            print('----------- Generation', generationIndex,'Complete----------')
            scores = np.array([pair[1] for pair in currentAgents])
            print('FITNESS = ', np.mean(scores))
            nextAgents = currentAgents[:configs.nSurvivors]
            currentAgents = mutateWeightsAndBiases(nextAgents, configs)
            saveWeightsAndBiases(nextAgents, generationIndex, configs)

        
            
            
            experiment.log_metric("fitness", np.mean(scores) , step= generationIndex)

