import numpy as np
import torch
import time
from agent import Agent
from snake_game import SnakeGame
import torch as T

env = SnakeGame(size_x=10, size_y=10)

n_games = 5000
agent = Agent(n_actions=4, gamma=0.95, epsilon=0.8, lr=0.001, input_dims=(1, 10, 10),
              mem_size=5000, eps_min=0.01, batch_size=64, replace=1000, eps_dec=1e-5)

n_steps = 0
scores = []
eps_history = []
counter_while = 0
for i in range(n_games):
    print(i)
    done = False
    score = 0
    obs = env.reset()
    while not done:
        # print("counter_whilee==",counter_while)
        action = agent.choose_action(obs)
        # print(obs)
        obs_, reward, done = env.make_step(action)
        score += reward
        agent.store_transition(obs_, action, reward, obs_, done)
        agent.learn()
        counter_while += 1
        if env.score>3:
            env.draw_board()
    print("score==", score,env.score)
    scores.append(score)
