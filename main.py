from comet_ml import Experiment
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import Args, configure
from API_KEYS import api_key, project_name
import torch

configs, use_cuda,  device = configure()

## SET LOGGING
# experiment = Experiment(project_name = project_name,  api_key = api_key)
# experiment.log_parameters(configs.getParamsDict())
    

def getTrainTest( isTest = False, experiment = None,):
    if isTest:
        return experiment.test()
    return experiment.train()


def mutateWeightsAndBiases(agents, configs:Args):
    nextAgents = []
    for i in range(configs.nAgents):
        pair = agents[np.random.randint(configs.nSurvivors)]
        agent = Agent(configs, device, stateDict = pair[0])
        for param in agent.parameters():
            param.data += configs.mutationPower * torch.randn_like(param)
        nextAgents.append(agent.getParams())
    return nextAgents





if __name__ == "__main__":
    
    
    env = Env(configs)
    
    currentAgents = []
    for spawnIndex in range(configs.nAgents):
        agent = Agent(configs, device)
        currentAgents.append(agent.getParams())


    nextAgents = []

    # with getTrainTest(configs.test, experiment):


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
                        
                        # print("Dead at score = ", round(score, 2), ' || Timesteps = ', t, ' || Reason = ', reasonForDeath)
                        break
                
                scores.append(score)
            
            avgScore = np.mean(np.array(scores))
            print('Generation = ', generationIndex, 'Spawn =', spawnIndex, 'Average score =',  avgScore)
            currentAgents.append([agent.getParams(), avgScore])


        

        currentAgents = sorted(currentAgents, key = lambda agent: agent[1], reverse = True)
        print('----------- Generation', generationIndex,'Complete----------')
        scores = np.array([pair[0] for pair in currentAgents])
        print('FITNESS = ', np.mean(scores))
        nextAgents = currentAgents[:configs.nSurvivors]
        currentAgents = mutateWeightsAndBiases(nextAgents, configs)

       
        
        
        # experiment.log_metric("scores", score , step= episodeIndex)

