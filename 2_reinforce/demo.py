import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
# % matplotlib inline

# avoid the gym warning message
gym.logger.set_level(40)

env = gym.make('MountainCar-v0')
env = env.unwrapped
# env.seed(1)

np.random.seed(1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
eps = np.finfo(np.float32).eps.item()


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=24, learning_rate=0.01):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def choose_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        act_probs = self.forward(state)
        c = Categorical(act_probs)
        action = c.sample()
        return action.item(), c.log_prob(action)

    def update_policy(self, vts, log_probs):
        policy_loss = []
        for log_prob, vt in zip(log_probs, vts):
            policy_loss.append(-log_prob * vt)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def discounted_norm_rewards(self, rewards, GAMMA):
        vt = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * GAMMA + rewards[t]
            vt[t] = running_add

        # normalized discounted rewards
        vt = (vt - np.mean(vt)) / (np.std(vt) + eps)
        return vt


policy_net = PolicyNetwork(state_space, action_space)


def main(episodes=5000, GAMMA=0.99):
    all_rewards = []
    running_rewards = []
    for episode in range(episodes):
        state = env.reset()
        rewards = []
        log_probs = []
        i = 0
        while True:
            i += 1
            action, log_prob = policy_net.choose_action(state)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                vt = policy_net.discounted_norm_rewards(rewards, GAMMA)
                policy_net.update_policy(vt, log_probs)
                all_rewards.append(np.sum(rewards))
                running_rewards.append(np.mean(all_rewards[-30:]))
                print("episode={},循环{}次时的状态:{}".format(episode, i, state))
                break
            state = new_state
        print('episode:', episode, 'total reward: ', all_rewards[-1], 'running reward:', int(running_rewards[-1]))
    return all_rewards, running_rewards, vt


all_rewards, running_rewards, vt = main(episodes=100)
# 下面将神经网络保存
# torch.save(policyNet, "policyNet.pkl")
