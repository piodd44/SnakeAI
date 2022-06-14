import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from snake_game import SnakeGame
from replay_memory import ReplayBuffer


class AgentQNet:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                 batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None):
        self.replace = replace
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.env_name = env_name
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        print(input_dims)
        self.q_eval = Net(lr=lr, n_action=n_actions, input_dims=input_dims)
        self.q_next = Net(lr=lr, n_action=n_actions, input_dims=input_dims)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.q_next.device)
            actions = self.q_eval(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        # print(states[0])
        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_).max(dim=1)[0]  # bo interesuje nas wartośc nie indesx

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class Net(nn.Module):
    def __init__(self, lr, input_dims, n_action):
        super(Net, self).__init__()
        print(input_dims)
        self.linear1 = nn.Linear(*input_dims, 512)
        self.linear2 = nn.Linear(512, n_action)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        actions = self.linear2(x)
        return actions


agent = AgentQNet(gamma=0.95, epsilon=1, lr=0.001, n_actions=4, input_dims=[7], mem_size=10 * 1000, batch_size=64,
                  eps_min=0.01, eps_dec=0.00001, replace=1000)

env = SnakeGame(10, 10)
learning_game = 10 * 1000
n_steps = 0
scores = []
eps_history = []
counter_while = 0
for i in range(learning_game):
    print(i)
    done = False
    score = 0
    obs = env.reset()
    obs = env.get_simple_state()
    while not done:
        # print("counter_whilee==",counter_while)
        action = agent.choose_action(obs)
        # print(obs)
        obs_, reward, done = env.make_step(action)
        obs_ = env.get_simple_state()
        score += reward
        agent.store_transition(obs_, action, reward, obs_, done)
        agent.learn()
        counter_while += 1
        if env.score > 3:
            env.draw_board()
    print("score==", env.score, score, agent.epsilon)
    scores.append(score)
