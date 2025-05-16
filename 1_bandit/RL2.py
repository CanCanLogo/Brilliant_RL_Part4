import numpy as np
import MDP
from sympy import *
import random
import math

class RL2:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward  # 传入bernoulli

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''
        # 根据伯努利分布给与reward
        reward = self.sampleReward(self.mdp.R[action, state])
        # 构造累积概率分布
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        # 用（0，1）随机数来与概率分布对比
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''从随机策略中采样单个动作的程序，通过以下概率公式采样
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        本函数将被reinforce()调用来选取动作

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action

        提示：计算出概率后，可以用np.random.choice()，来进行采样
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded

        # 计算softmax概率分布
        # print(policyParams[:, state])
        # print(state)
        # print(policyParams[:, state])
        vec = policyParams[:, state] - np.max(policyParams[:, state])
        # print(vec)
        exp_values = np.exp(vec)

        probabilities = exp_values / np.sum(exp_values)

        # 根据概率分布采样动作
        action = np.random.choice(len(probabilities), p=probabilities)

        return action, probabilities



    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy 算法 for bandits (假设没有折扣因子).
        Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        '''
        共有|A|个老虎机，可以选择A个动作
        状态只有一个，不用考虑
        '''
        empiricalMeans = np.zeros(self.mdp.nActions)
        empiricalNum = np.ones(self.mdp.nActions)  # 为了防止除以0问题
        empiricalReward = np.zeros(self.mdp.nActions)
        reward_list = []
        # 随机初始化state
        state = random.randint(0, self.mdp.nStates - 1)
        # state固定

        for i in range(nIterations):
            epsilon = 1.0 / (i + 1)
            # 类似模拟退火思想，epsilon逐渐下降，加一防止除以0
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(self.mdp.nActions)
                # exploration : random action selection
            else:
                action = np.argmax(np.array(empiricalMeans))
                # exploitation: optimal action selection
            # 生成reward
            # print(action)
            reward, _ = self.sampleRewardAndNextState(state, action)
            reward_list.append(reward)  # record the reward

            # update the empirical means based on the observed reward
            empiricalNum[action] += 1
            empiricalReward[action] += reward
            empiricalMeans[action] = empiricalReward[action] / empiricalNum[action]
            # empiricalMeans[action] += (reward - empiricalMeans[action]) / len(reward_list)
            # state = nextState  arm问题无需nextstate
        return empiricalMeans, reward_list

    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling 算法 for Bernoulli bandits (假设没有折扣因子)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0]
                is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
                alpha和beta分别代表在a+b次伯努利试验中成功和失败的次数。
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards 参数k表示采样次数。默认值为1，即每次迭代只进行一次采样

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)

        提示：根据beta分布的参数，可以采用np.random.beta()进行采样
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        # 为了计算mean
        empiricalNum = np.ones(self.mdp.nActions)  # 为了防止除以0问题
        empiricalReward = np.zeros(self.mdp.nActions)

        state = random.randint(0, self.mdp.nStates - 1)
        # state固定
        reward_list = []

        for i in range(nIterations):
            samples = np.zeros((self.mdp.nActions, k))
            # 记录采样结果
            for a in range(self.mdp.nActions):
                for j in range(k):
                    # k是采样次数
                    samples[a, j] = np.random.beta(prior[a, 0], prior[a, 1])
            action = np.argmax(np.mean(samples, axis=1))

            reward, _ = self.sampleRewardAndNextState(state, action)
            reward_list.append(reward)  # record the reward

            if reward > 0.5:
                prior[action, 0] += 1
            elif reward < 0.5:
                prior[action, 1] += 1

            # 更新平均收益表
            empiricalNum[action] += 1
            empiricalReward[action] += reward
            empiricalMeans[action] = empiricalReward[action] / empiricalNum[action]

        return empiricalMeans,reward_list

    def UCBbandit(self, nIterations):
        '''Upper confidence bound 算法 for bandits (假设没有折扣因子)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        # 为了计算mean
        empiricalNum = np.ones(self.mdp.nActions)  # 为了防止除以0问题
        empiricalReward = np.zeros(self.mdp.nActions)

        state = random.randint(0, self.mdp.nStates - 1)
        # state固定
        reward_list = []
        K = 10
        for i in range(nIterations):
            if i < K:
                action = np.random.choice(self.mdp.nActions)
            else:
                ucb_values = empiricalMeans + np.sqrt(2 * np.log(i + 1) / empiricalNum)
                action = np.argmax(ucb_values)

            reward, _ = self.sampleRewardAndNextState(state, action)
            reward_list.append(reward)  # record the reward

            # empiricalMeans[action] += (reward - empiricalMeans[action]) / (i + 1)
            # 更新平均收益表
            empiricalNum[action] += 1
            empiricalReward[action] += reward
            empiricalMeans[action] = empiricalReward[action] / empiricalNum[action]

        return empiricalMeans,reward_list

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        '''reinforce 算法，学习到一个随机策略，建模为：
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        上面的sampleSoftmaxPolicy()实现该方法，通过调用sampleSoftmaxPolicy(policyParams,state)来选择动作
        并且同学们需要根据上课讲述的REINFORCE算法，计算梯度，根据更新公式，完成策略参数的更新。
        其中，超参数：折扣因子gamma=0.95，学习率alpha=0.01

        Inputs:
        s0 -- 初始状态
        initialPolicyParams -- 初始策略的参数 (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- 最终策略的参数 (array of |A|x|S| entries)
        rewardList --用于记录每个episodes的累计折扣奖励 (array of |nEpisodes| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded

        # policyParams = np.zeros((self.mdp.nActions, self.mdp.nStates))

        policyParams = initialPolicyParams.copy()
        rewardList = []
        alpha = 0.01
        gamma = 0.95

        for episode in range(nEpisodes):
            # Initialize state and action list
            state = s0
            states = [state]
            actions = []
            reward_episode = []
            G_list = []
            probility = []
            reward_add_return = 0

            for s in range(self.mdp.nStates):
                _, prob = self.sampleSoftmaxPolicy(policyParams, s)
                probility.append(prob)
            # print(probility)

            # 每一个episode先做step个行动然后再更新
            for step in range(nSteps):
                # Sample an action from the policy
                action, _ = self.sampleSoftmaxPolicy(policyParams, state)
                # action = np.random.choice(len(probility[state]), p=probility[state])
                actions.append(action)
                # Take the action and observe the next state and reward
                reward, nextState = self.sampleRewardAndNextState(state, action)
                reward_episode.append(reward)
                reward_add_return += reward

                # Update the state and add it to the state list
                state = nextState
                states.append(state)
                # Update the reward list
                if step == 0:
                    rewardList.append(reward)
                else:
                    rewardList[episode] += (gamma ** step) * reward
            # rewardList.append(reward_add_return)

            for t in range(0, nSteps-1):
                running_add = 0
                for k in range(t, nSteps):
                    running_add += gamma ** (k-t) * reward_episode[k]
                G_list.append(running_add)
            for s in range(self.mdp.nStates):
                _, prob = self.sampleSoftmaxPolicy(policyParams, s)
                probility.append(prob)
            probility = np.transpose(probility)
            # Update the policy parameters
            for t in range(nSteps - 1):
                # Compute the gradient of the log-probability of the action
                gradLogProb = np.zeros((self.mdp.nActions, self.mdp.nStates))
                for a in range(self.mdp.nActions):
                    for s in range(self.mdp.nStates):
                        if a == actions[t] and s == states[t]:
                            gradLogProb[a, s] = 1 - probility[a, s]
                        # else:
                        #     gradLogProb[a, s] = - probility[a, s]
                        # softmax和cross entropy结合的求导过程最终就是pred_i - label_i
                policyParams += alpha * (gamma ** t) * gradLogProb * G_list[t]
        return [policyParams, rewardList]

    # 其实最后正确的方法都已经尝试过，但是问题就是：
    # 一 应该使用概率值进入计算而不是policy值
    # 二 应该只对正确的进行奖励，而其他的不应处罚
    # 三 计算probility出现了失误s,state

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon=0, temperature=0):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Boltzmann exploration:softmax 对于自己有把握的状态，就应该采取exploitation；而对于自己没有把握的状态，由于训练中的输赢不重要，
        所以可以多去尝试exploration，但同时也不是盲目地乱走，而是尝试一些自己认为或许还不错的走法。
        https://zhuanlan.zhihu.com/p/166410986

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs:
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''
        temperature_delta = 1
        epsilon_rate = 0.99
        episode = 0
        Qtable = initialQ
        learning_rate = 0.05
        rewardList = []
        gamma = 0.95
        policy_history = []

        while episode < nEpisodes:
            s = s0
            episode += 1
            step = 0
            rewardSum = 0
            temperature += temperature_delta
            epsilon *= epsilon_rate
            while step < nSteps:
                step += 1
                # 获得s下a的各q值

                A_choose = Qtable[:, s]
                A_choose = A_choose.squeeze()
                # print(A_choose)

                # 生成备选的a列表
                actions = np.arange(0, 1, Qtable.shape[0])

                # linspase会出现float

                # 取a
                # a = 0
                eps = random.uniform(0, 1)
                # print(eps)
                if eps >= epsilon:
                    # exploitation
                    # softmax忘记除了
                    softmax = [temperature * math.exp(A_choose[i]) for i in range(Qtable.shape[0])]
                    softSum = sum(softmax)
                    softmax = [softmax[i] / softSum for i in range(Qtable.shape[0])]
                    # print(softmax)
                    cumProb = np.cumsum(softmax)  # 把
                    # print(cumProb)
                    # print(np.where(cumProb >= np.random.rand(1)))
                    a = np.where(cumProb >= np.random.rand(1))[0][0]
                else:
                    a = np.random.choice(actions)
                # print(a)
                vector = self.sampleRewardAndNextState(s, a)
                # (s,a,r,s')
                reward = vector[0]
                # print(reward)
                nextState = vector[1]

                # qtable_nextS = Qtable[:, nextState]
                # qtable_nextS = qtable_nextS.squeeze()
                qtable_nextS = Qtable[:, nextState]
                qtable_nextS = qtable_nextS.squeeze()

                current_q = Qtable[a][s]
                # 贝尔曼方程更新
                new_q = reward + self.mdp.discount * max(qtable_nextS)
                # print(current_q, new_q)
                Qtable[a][s] += learning_rate * (new_q - current_q)
                # print(reward)

                rewardSum += reward
                s = nextState
                if step == 1:
                    rewardList.append(reward)
                else:
                    rewardList[episode-1] += (gamma ** (step-1)) * reward
            # print(rewardSum)
            # rewardList.append(rewardSum)

        Q = Qtable
        policy = [np.argmax(Qtable[:, i].squeeze()) for i in range(Qtable.shape[1])]
        # rewardList = np.zeros(nEpisodes)

        return [Q, policy, rewardList]






        # # Initialize policy parameters
        # policyParams = initialPolicyParams
        #
        # # Initialize reward list
        # rewardList = []
        #
        # # Set hyperparameters
        # gamma = 0.95  # discount factor
        # alpha = 0.01  # learning rate
        #
        #
        # for episode in range(nEpisodes):
        #     # Initialize state and action list
        #     state = s0
        #     states = [state]
        #     actions = []
        #
        #     # Generate an episode
        #     for step in range(nSteps):
        #         # Sample an action from the policy
        #         action, _ = self.sampleSoftmaxPolicy(policyParams, state)
        #         actions.append(action)
        #
        #         # Take the action and observe the next state and reward
        #         reward, nextState= self.sampleRewardAndNextState(state, action)
        #
        #         # Update the state and add it to the state list
        #         state = nextState
        #         states.append(state)
        #
        #         # Update the reward list
        #         if step == 0:
        #             rewardList.append(reward)
        #         else:
        #             rewardList[episode] += gamma ** step * reward
        #
        #     # Update the policy parameters
        #     for t in range(nSteps):
        #         # Compute the gradient of the log-probability of the action
        #         gradLogProb = np.zeros((self.mdp.nActions, self.mdp.nStates))
        #         for a in range(self.mdp.nActions):
        #             for s in range(self.mdp.nStates):
        #                 if a == actions[t] and s == states[t]:
        #                     gradLogProb[a, s] = 1 - self.sampleSoftmaxPolicy(policyParams, s)[1][a]
        #                 else:
        #                     gradLogProb[a, s] = -self.sampleSoftmaxPolicy(policyParams, s)[1][a]
        #
        #         # Update the policy parameters using the REINFORCE update rule
        #         policyParams += alpha * gamma ** t * gradLogProb * rewardList[episode]
        #
        # return policyParams, rewardList

        # for episode in range(nEpisodes):
        #     print(episode)
        #     s = s0
        #     episode_reward = 0
        #     for step in range(nSteps):
        #         action = self.sampleSoftmaxPolicy(policyParams, s)
        #
        #         reward, nextState = self.sampleRewardAndNextState(s, action)
        #         episode_reward += reward
        #         G = 0  # G 是折扣累积回报（Discounted Cumulative Reward）
        #         discounted_rewards = []
        #         for t in range(step + 1):
        #             G = gamma  * G + reward
        #             discounted_rewards.append(G)
        #
        #         # 将列表 discounted_rewards 中的元素顺序反转
        #         discounted_rewards = discounted_rewards[::-1]
        #
        #         for t in range(len(discounted_rewards)):
        #             G = discounted_rewards[t]
        #             policyParams[action][s] += alpha * (G - policyParams[action][s])
        #         s = nextState
        #     rewardList.append(episode_reward)
