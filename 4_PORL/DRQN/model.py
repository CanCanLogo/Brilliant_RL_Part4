import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import gamma, device, batch_size

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        # The inputs are two integers giving the dimensions of the inputs and outputs respectively. 
        # The input dimension is the state dimention and the output dimension is the action dimension.
        # This constructor function initializes the network by creating the different layers. 
        # This function now only implements two fully connected layers. Modify this to include LSTM layer(s). 
        
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # self.fc1 = nn.Linear(self.num_inputs, 128)
        # self.lstm = nn.LSTM(input_size=self.num_inputs, hidden_size=128, batch_first=True)
        # self.fc2 = nn.Linear(128, num_outputs)
        # print(num_outputs)
        # 输入层到LSTM层的线性层
        self.fc1 = nn.Linear(self.num_inputs,16)
        # LSTM层
        # self.lstm = nn.LSTM(128, 16)
        self.lstm = nn.LSTM(input_size = 16, hidden_size = 16, num_layers = 1, batch_first=True)
        # LSTM层到输出层的线性层
        self.fc2 = nn.Linear(16, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)



    def forward(self, x, hidden=None):
        # The variable x denotes the input to the network. 
        # The hidden variable denotes the hidden state and cell state inputs to the LSTM based network. 
        # The function returns the q value and the output hidden variable information (new cell state and new hidden state) for the given input. 
        # This function now only uses the fully connected layers. Modify this to use the LSTM layer(s).          

        # out = F.relu(self.fc1(x))
        # qvalue = self.fc2(out)
        #
        # return qvalue, hidden

        # x = x.unsqueeze(0)  # Add batch dimension
        # out, hidden = self.lstm(x, hidden)
        # out = F.relu(out[:, -1, :])  # Use the last output of the sequence
        # qvalue = self.fc(out)
        #
        # return qvalue, hidden

        # 输入层到LSTM层的线性变换，然后通过ReLU激活函数

        out = F.relu(self.fc1(x))
        # 通过LSTM层，输出状态和细胞状态
        out, hidden = self.lstm(out.view(1, -1, 16), hidden)
        # LSTM层到输出层的线性变换，然后通过ReLU激活函数
        qvalue = F.relu(self.fc2(out.view(1, -1)))
        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        # The online_net is the variable that represents the first (current) Q network.
        # The target_net is the variable that represents the second (target) Q network.
        # The optimizer is Adam. 
        # Batch represents a mini-batch of memory. Note that the minibatch also includes the rnn state (hidden state) for the DRQN. 

        # This function takes in a mini-batch of memory, calculates the loss and trains the online network. Target network is not trained using back prop. 
        # The loss function is the mean squared TD error that takes the difference between the current q and the target q. 
        # Return the value of loss for logging purposes (optional).

        # Implement this function. Currently, temporary values to ensure that the program compiles. 

        # loss = 1.0
        #
        # return loss
        # 获取在线网络和目标网络的参数，用于计算损失和优化网络参数

        states, actions, rewards, next_states, dones, hidden = batch

        # 获取批量数据的大小
        batch_size = len(states)
        print(batch_size)
        # 使用在线网络预测当前状态的Q值
        # print(len(batch))
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        # masks = torch.FloatTensor(masks).to(device)
        q_values = online_net(states, hidden)

        # 根据动作选择对应的Q值
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 使用目标网络预测下一个状态的Q值，并取最大值作为目标Q值
        next_q_values = target_net(next_states, hidden).max(1)[0].detach()

        # 计算目标Q值，如果done为True，则使用奖励值作为目标Q值
        target_q_value = rewards + (1 - dones) * next_q_values

        # 计算损失函数，使用均方误差损失函数（MSE Loss）
        loss = nn.MSELoss()(q_value, target_q_value.unsqueeze(1))

        # 反向传播计算梯度，并更新在线网络的参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 返回损失值用于日志记录（可选）
        return loss.item()
    
    
    def get_action(self, state, hidden):
        # state represents the state variable. 
        # hidden represents the hidden state and cell state for the LSTM.
        # This function obtains the action from the DRQN. The q value needs to be obtained from the forward function and then a max needs to be computed to obtain the action from the Q values. 
        # Implement this function. 
        # Template code just returning a random action.

        # action = 0
        # return action, hidden
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_value, hidden = self.forward(state, hidden)
        action = q_value.max(1)[1].item()
        return action, hidden