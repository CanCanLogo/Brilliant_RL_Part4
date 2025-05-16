import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.autograd import Variable
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

import torch.nn.functional as F


class ExpertTraj:
    """
    class holding the expert trajectories
    """
    def __init__(self, env_name):
        self.exp_states = np.loadtxt("./expert_traj/{}/{}_expert_states.dat".format(env_name, env_name))
        self.exp_actions = np.loadtxt("./expert_traj/{}/{}_expert_actions.dat".format(env_name, env_name))
        self.n_transitions = len(self.exp_actions)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)


class Actor(nn.Module):
    """
    Actor, policy function
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x


class Discriminator(nn.Module):
    """
    Discriminator, act like a value function
    """
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


class GAIL:

    def __init__(self, args, env_name, log_file):
        self.env = gym.make(args.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        self.args = args
        self.device = args.device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.discriminator = Discriminator(state_dim, action_dim).to(self.device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001,
                                                    betas=(0.5,0.999))
        self.max_action = max_action
        self.expert = ExpertTraj(env_name)
        self.loss_fn = nn.BCELoss()

        self.log_file = log_file
        self.rng = np.random.RandomState()

        self.gamma = 0.95
        self.clip_ratio = 0.2

    def select_action(self, state):
        """
        actor selects the action
        :param state: game state
        :return: continuous actions
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def plot(self, batch_nums, perf_nums, y_tag):
        """
        Draw the plot
        :param batch_nums: x-axis numbers
        :param perf_nums: y-axis numbers
        :param y_tag: y-axis description
        """
        plt.figure(figsize=(20,5))
        plt.xlabel('Training Step', fontsize=15)
        plt.ylabel(y_tag, fontsize=15)
        plt.plot(batch_nums, perf_nums)
        plt.savefig('gail_{0}.png'.format(y_tag))

    def test(self, max_timesteps=1500):
        """
        testing the actor
        :param max_timesteps: max testing step
        :return: total testing rewards
        """
        totalr = 0.
        state = self.env.reset()
        for t in range(max_timesteps):
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            totalr += reward
            if done:
                break
        print('Total reward is {0}'.format(totalr), file=self.log_file, flush=True)
        return totalr

    def train(self, n_iter):
        """
        training GAIL
        :param n_iter: the number of training steps
        """

        d_running_loss = 0
        a_running_loss = 0
        training_rewards_record = []
        training_d_loss_record = []
        training_a_loss_record = []
        training_steps_record = []

        for train_step in range(n_iter + 1):

            # sample expert transitions
            exp_state, exp_action = self.expert.sample(self.args.batch_size)
            exp_state = torch.FloatTensor(exp_state).to(self.device)
            exp_action = torch.FloatTensor(exp_action).to(self.device)

            # sample expert states for actor
            state, _ = self.expert.sample(self.args.batch_size)
            state = torch.FloatTensor(state).to(self.device)

            # probs = self.actor.forward(state)
            # m = torch.distributions.Categorical(probs)
            # action = m.sample()
            # logp = m.log_prob(action)  # 根据m这个多项式分布，计算act的对数概率值

            # mu = self.actor(state)
            # # std = self.action_std * torch.ones_like(mu)
            # init_std = 1
            # std = init_std * torch.ones_like(mu)
            # dist = torch.distributions.Normal(mu, std)
            # action = dist.sample()
            # logp_old = dist.log_prob(action)

            action = self.actor.forward(state)

            # state_next, reward, done, _ = self.env.step(action)

            q_value = self.discriminator(state, action)
            q_value_tar = self.discriminator(exp_state, exp_action)

            loss_expert = torch.mean(torch.log(1 - torch.clamp(q_value_tar, min=0.01, max=1)))
            # loss_exp = self.loss_fn(q_value_tar, torch.zeros_like(q_value_tar))
            loss_agent = torch.mean(torch.log(torch.clamp(q_value, min=0.01, max=1)))
            # loss_ac = self.loss_fn(q_value, torch.zeros_like(q_value))
            # entropy_loss = F.binary_cross_entropy_with_logits(q_value, torch.zeros_like(q_value), reduction='mean')
            # print(q_value_tar)
            # print(q_value)
            expert_loss = F.binary_cross_entropy_with_logits(q_value_tar,
                                                             torch.ones(q_value_tar.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(q_value,
                                                             torch.zeros(q_value.size()).to(self.device))
            loss_discriminator = expert_loss + policy_loss
            # print(loss_discriminator)
            # loss_discriminator = -loss_discriminator

            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()

            # 以下是填空部分，以下两行需要用实际的loss和gradient step来替换，现在的这两行是占位符来确保程序可编译


            # loss_discriminator = self.loss_fn(exp_action.float(), action.float())
            # loss_discriminator.backward()
            # self.optim_discriminator.step()

            # loss_discriminator = torch.zeros([1]).to(self.device)  # write the discriminator loss
            # loss_discriminator = Variable(loss_discriminator, requires_grad=True)
            loss_discriminator.backward()
            self.optim_discriminator.step()

            ################
            # update policy
            ################
            self.optim_actor.zero_grad()
            # # 以下是填空部分，以下两行需要用实际的loss和gradient step来替换，现在的这两行是占位符来确保程序可编译
            # # loss_actor = torch.zeros([1]).to(self.device)  # write the actor loss
            # # loss_discriminator = Variable(loss_discriminator, requires_grad=True)
            # should_action = exp_action.view(-1)
            # we_action = self.actor.forward(exp_state).view(-1)
            # # print(should_action)
            # # print(we_action)
            # criterion = nn.MSELoss()
            # loss_actor = criterion(should_action, we_action)

            # q = self.discriminator(state, action)
            # loss_actor = -q

            # loss_actor = criterion(torch.mean(q).view(-1)[0], torch.tensor(0).float().to(self.device))
            # print(loss_actor)

            # 计算价值函数



            # mu = self.actor(exp_state)
            # # std = self.action_std * torch.ones_like(mu)
            # init_std = 1
            # std = init_std * torch.ones_like(mu)
            # dist = torch.distributions.Normal(mu, std)
            # logprob_policy = dist.log_prob()
            mu = self.actor(exp_state)
            # std = self.action_std * torch.ones_like(mu)
            init_std = 1
            std = init_std * torch.ones_like(mu)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action)

            policy_d = self.discriminator(state, action).squeeze()
            score = torch.sigmoid(policy_d)
            gail_rewards = score.log() - (1 - score).log()


            returns = torch.zeros_like(gail_rewards)
            for t in reversed(range(len(gail_rewards))):
                if t == len(gail_rewards) - 1:
                    returns[t] = gail_rewards[t]
                else:
                    returns[t] = gail_rewards[t] + self.gamma * returns[t + 1]
            '''
           因为这里是actor-only PPO，所以并没有对应的价值函数估计。
           所以，我们这里用的是policy输出的动作概率分布，它不能被理解为状态的价值函数估计。
           但是，它可以用来计算优势估计并作为PPO损失函数的动作概率分布项。
            '''
            values = self.actor(state)
            adv = returns - torch.mean(values, axis=1)
            # adv = returns
            # print(adv)
            # 计算旧策略的动作概率和对数概率

            # print(exp_action)
            # print(torch.argmax(exp_action, dim = 1))
            # pi_old = self.actor(state).gather(torch.argmax(exp_action, dim = 1)).squeeze(-1)  # 得到旧策略的动作概率
            # logp_old = dist.log_prob(exp_action)
            ratio = torch.mean(torch.exp(logp - logp))  # 通过log性质用减法计算比率
            # ratio = torch.ones_like(gail_rewards)
            surr1 = ratio * adv  # 第一项损失
            # print(surr1)
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv  # 第二项损失
            # surr2 = adv
            loss_actor = torch.min(surr1, surr2).mean()  # PPO损失函数
            # loss_actor = torch.mean(adv)
            # loss_actor = -torch.mean(q_value) + loss_discriminator
            # loss_actor = -gail_rewards
            loss_actor.backward(retain_graph = True)
            # retain_graph = True
            # loss_actor.backward()
            self.optim_actor.step()
            d_running_loss += loss_discriminator.item()
            a_running_loss += loss_actor.mean().item()

            # state = state_next

            values = self.actor(state)
            adv = returns - torch.mean(values, axis=1)
            ratio = torch.mean(torch.exp(logp - logp))
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            loss_actor = torch.min(surr1, surr2).mean()
            if train_step % 100 == 0:
                print('[%d] discriminator loss: %.6f, actor loss %.6f' % (train_step + 1,
                                                                          d_running_loss / (train_step + 1),
                                                                          a_running_loss / (train_step + 1)),
                      file=self.log_file, flush=True)
                totalr = self.test()
                training_rewards_record.append(totalr)
                training_steps_record.append(train_step)
                training_d_loss_record.append(loss_discriminator.item())
                training_a_loss_record.append(loss_actor.mean().item())

        avg_last_10_rewards = []
        for idx in range(len(training_rewards_record)):
            if idx >= 10:
                avg_last_10_rewards.append(np.mean(training_rewards_record[idx - 9:idx + 1]))
            else:
                avg_last_10_rewards.append(np.mean(training_rewards_record[:idx + 1]))

        self.plot(batch_nums=training_steps_record, perf_nums=avg_last_10_rewards, y_tag='Rewards')
        self.plot(batch_nums=training_steps_record, perf_nums=training_d_loss_record, y_tag='Discriminator_Loss')
        self.plot(batch_nums=training_steps_record, perf_nums=training_a_loss_record, y_tag='Actor_Loss')


class BehaviorCloning:

    def __init__(self, args, env_name, log_file):
        self.env = gym.make(args.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        self.args = args
        self.device = args.device
        self.model = Actor(state_dim, action_dim, max_action).to(self.device)
        self.optim_actor = torch.optim.Adam(self.model.parameters(), lr=0.00003)

        self.log_file = log_file
        self.rng = np.random.RandomState()
        self.expert = ExpertTraj(env_name)

    def train(self, n_iter):
        """
        training Behavior Cloning
        :param n_iter: the number of training steps
        """
        criterion = nn.MSELoss()

        running_loss = 0
        training_rewards_record = []
        training_loss_record = []
        training_steps_record = []
        for train_step in (range(n_iter + 1)):
            exp_state, exp_action = self.expert.sample(self.args.batch_size)
            # print(exp_action)
            exp_state = torch.FloatTensor(exp_state).to(self.device)
            exp_action = torch.FloatTensor(exp_action).to(self.device)
            # print(exp_action)
            outputs = self.model(exp_state)
            # print(outputs)
            #填写BC的loss函数(一行)，训练模型
            # criterion = nn.CrossEntropyLoss()
            loss = criterion(exp_action, outputs)

            # states = torch.tensor(states, dtype=torch.float32)
            # actions = torch.tensor(actions).view(-1, 1)
            # print(outputs.shape)
            # print(torch.argmax(exp_action, dim = 1).shape)
            # log_probs = torch.log(-torch.gather(outputs, 1, torch.argmax(exp_action, dim = 1).view(-1, 1)))
            # loss = torch.mean(log_probs)  # 最大似然估计
            
            self.optim_actor.zero_grad()
            loss.backward()
            self.optim_actor.step()
            running_loss += loss.item()
            if train_step % 100 == 0:
                print('[%d] loss: %.6f' % (train_step + 1, running_loss / (train_step + 1)), file=self.log_file,
                      flush=True)
                totalr = self.test()
                training_rewards_record.append(totalr)
                training_loss_record.append(loss.item())
                training_steps_record.append(train_step)
        avg_last_10_rewards = []
        for idx in range(len(training_rewards_record)):
            if idx >= 10:
                avg_last_10_rewards.append(np.mean(training_rewards_record[idx - 9:idx + 1]))
            else:
                avg_last_10_rewards.append(np.mean(training_rewards_record[:idx + 1]))

        self.plot(batch_nums=training_steps_record, perf_nums=avg_last_10_rewards, y_tag='Rewards')
        self.plot(batch_nums=training_steps_record, perf_nums=training_loss_record, y_tag='Loss')

    def select_action(self, state):
        """
        actor selects the action
        :param state: game state
        :return: continuous actions
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.model(state).cpu().data.numpy().flatten()

    def plot(self, batch_nums, perf_nums, y_tag):
        """
        Draw the plot
        :param batch_nums: x-axis numbers
        :param perf_nums: y-axis numbers
        :param y_tag: y-axis description
        """
        plt.figure(figsize=(20,5))
        plt.xlabel('Training Step', fontsize=15)
        plt.ylabel(y_tag, fontsize=15)
        plt.plot(batch_nums, perf_nums)
        plt.savefig('behavior_cloning_{0}.png'.format(y_tag))

    def test(self, max_timesteps=1500):
        """
        testing the actor
        :param max_timesteps: max testing step
        :return: total testing rewards
        """
        totalr = 0.
        state = self.env.reset()
        for t in range(max_timesteps):
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            totalr += reward
            if done:
                break
        print('Total reward is {0}'.format(totalr), file=self.log_file, flush=True)
        return totalr


def gail(args):
    """
    run GAIL
    :param args: parameters
    """
    if args.log_dir is not None:
        log_file = open(args.log_dir, 'w')
    else:
        log_file = None
    student = GAIL(args, args.env_name, log_file)
    student.train(n_iter=50000)


def behavior_cloning(args):
    """
    run behavior cloning
    :param args: parameters
    """
    if args.log_dir is not None:
        log_file = open(args.log_dir, 'w')
    else:
        log_file = None
    student = BehaviorCloning(args, args.env_name, log_file)
    student.train(n_iter=50000)


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=None, type=str, help='the log file')
    parser.add_argument('--mode',
                        choices=['cloning', 'gail'],
                        help='Learning mode')
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='The name of device')
    parser.add_argument('--env_name', type=str, default='BipedalWalker-v2')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--expert_model_path', type=str,
                        default='./expert_model/highway-DQN-expert-baselines-2021-03-14.pt')

    args = parser.parse_args()
    return args


def main():
    args = init_config()
    args.mode = 'cloning'
    if args.mode == 'cloning':
        behavior_cloning(args)
    elif args.mode == 'gail':
        gail(args)
    else:
        raise ValueError("Unknown running mode: {0}".format(args.mode))


if __name__ == '__main__':
    main()
