from config import Args
from neuralnet import Net
import torch


class Agent():
    def __init__(self, args:Args, device, stateDict = None):

        self.args = args
        self.net = Net(args).double().to(device)
        self.device = device
        if stateDict != None:
            self.net.load_state_dict(stateDict)

    def chooseAction(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.net(state)
        return action.squeeze().cpu().numpy()

    def save(self, episode ):
        self.lastSavedEpisode = episode
        print('-----------------------------------------')
        print("SAVING at ", episode)
        print('-----------------------------------------')
        torch.save(self.net.state_dict(), self.args.saveLocation + 'episode-' + str(episode) +  '.pkl')
    
    def getParams(self):
        return self.net.state_dict()

        

    # def update(self, transition, episodeIndex):
    #     self.buffer[self.counter] = transition
    #     self.counter += 1
    #     if self.counter == self.buffer_capacity:
    #         print("UPDATING WEIGHTS AT EPISODE = ", episodeIndex)
    #         self.counter = 0
    #         self.training_step += 1

    #         s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
    #         a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
    #         r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
    #         s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

    #         old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

    #         with torch.no_grad():
    #             target_v = r + self.args.gamma * self.net(s_)[1]
    #             advantage = target_v - self.net(s)[1]

    #         for _ in range(self.ppo_epoch):
    #             for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

    #                 alpha, beta = self.net(s[index])[0]
    #                 dist = Beta(alpha, beta)
    #                 a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
    #                 ratio = torch.exp(a_logp - old_a_logp[index])

    #                 surr1 = ratio * advantage[index]
    #                 surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage[index]
    #                 actorLoss = -torch.min(surr1, surr2).mean()
    #                 criticLoss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
    #                 loss = actorLoss + 2. * criticLoss

    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
    #                 self.optimizer.step()
    #         if episodeIndex - self.prevSaveIndex > 10:
    #             self.save(episodeIndex)
    #             self.prevSaveIndex = episodeIndex