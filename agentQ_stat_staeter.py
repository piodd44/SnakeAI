from snake_game import SnakeGame
from agentQ_stat import AgentQStat
import time
import numpy as np

snake_game = SnakeGame(10, 10)
agent = AgentQStat(4)
random_learning_game = 10 * 1000
epsilon_learning_game = 4 * 1000
for x in range(random_learning_game):
    print(x)
    obs = snake_game.reset()
    obs = snake_game.get_simple_state()
    done = False
    sum_of_reward = 0
    while not done:
        move = agent.random_move(obs)
        next_obs, reward, done = snake_game.make_step(move)
        next_obs = snake_game.get_simple_state()
        sum_of_reward += reward
        obs = next_obs
        agent.last_state_reward(reward)
        agent.last_state_reset()
agent.update_the_best_move()
score_list = []
if True:
    for x in range(epsilon_learning_game):
        # print("epsilon ,", x, "epsilon==", agent.epsilon)
        obs = snake_game.reset()
        obs = snake_game.get_simple_state()
        done = False
        sum_of_reward = 0
        score=0
        while not done:
            score=snake_game.score
            move = agent.epsilon_move(obs)
            next_obs, reward, done = snake_game.make_step(move)
            next_obs = snake_game.get_simple_state()
            sum_of_reward += reward
            obs = next_obs
            agent.last_state_reward(reward)
            agent.last_state_reset()
            if x > 1500:
                snake_game.draw_board()

        score_list.append(score)
        if x % 100 == 0:
            print("w pętli")
            print(np.mean(score_list[-100:]))
            # agent.update_the_best_move()

for x in range(100000):
    print(x)
    obs = snake_game.reset()
    obs = snake_game.get_simple_state()
    done = False
    sum_of_reward = 0
    while not done:
        move = agent.the_best_move(obs)
        next_obs, reward, done = snake_game.make_step(move)
        snake_game.draw_board()
        time.sleep(0.1)
        next_obs = snake_game.get_simple_state()
        sum_of_reward += reward
        obs = next_obs
    # agent.update_reward(sum_of_reward)
    agent.reset()
