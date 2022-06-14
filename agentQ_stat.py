import random

import numpy as np

from snake_game import SnakeGame
import time
import numpy


class AgentQStat:
    def __init__(self, nr_of_actions):
        self.state_dictionary = {}
        self.nr_of_actions = nr_of_actions
        self.nr_of_states = 0
        self.epsilon = 1
        self.last_state = None
        self.min_epsilon = 0.01

    def epsilon_move(self, state):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= 0.00001
        if random.random() > self.epsilon:
            return self.the_best_move(state)
        else:
            return self.random_move(state)

    def random_move(self, state):
        if state in self.state_dictionary:
            pass
        else:
            self.nr_of_states += 1
            self.state_dictionary[state] = State(nr_of_action=self.nr_of_actions)
        self.last_state = self.state_dictionary[state]
        move = self.state_dictionary[state].random_move()
        return move

    def the_best_move(self, state):
        if state in self.state_dictionary:
            pass
        else:
            self.state_dictionary[state] = State(nr_of_action=self.nr_of_actions)
        self.last_state = self.state_dictionary[state]
        move = self.state_dictionary[state].get_best_move()
        return move

    def reset(self):
        for state in self.state_dictionary:
            self.state_dictionary[state].reset()

    def update_the_best_move(self):
        print("liczba stanów ==", self.nr_of_states)
        for state in self.state_dictionary:
            self.state_dictionary[state].update_statistics()

    def update_reward(self, reward):
        for state in self.state_dictionary:
            self.state_dictionary[state].update_reward(reward)

    def last_state_reward(self, reward):
        self.last_state.update_reward(reward)

    def last_state_reset(self):
        self.last_state.reset()


class State:
    def __init__(self, nr_of_action):
        self.nr_of_action = nr_of_action
        self.action_dictionary = {}
        self.chosen_dictionary = {}
        self.move_statistics_dictionary = {}
        for i in range(nr_of_action):
            self.action_dictionary[i] = [1.0, 0.0]
            self.chosen_dictionary[i] = False
            self.move_statistics_dictionary[i] = 1.0
        self.the_best_move = 0

    def reset(self):
        for i in self.chosen_dictionary:
            self.chosen_dictionary[i] = False

    def update_reward(self, reward):
        for i in self.action_dictionary:
            if self.chosen_dictionary[i]:
                self.action_dictionary[i][1] += reward

    def random_move(self):
        move = random.randint(0, self.nr_of_action - 1)
        if not self.chosen_dictionary[move]:
            self.chosen_dictionary[move] = True
            self.action_dictionary[move][0] += 1
        return move

    def update_statistics(self):
        the_best = -100000000
        for move in self.action_dictionary:
            all_moves = self.action_dictionary[move][0]
            all_reward = self.action_dictionary[move][1]
            avg = all_reward / all_moves
            if avg > the_best:
                the_best = avg
                self.the_best_move = move
            self.move_statistics_dictionary[move] = avg

    def get_best_move(self):
        move = self.the_best_move
        if not self.chosen_dictionary[move]:
            self.chosen_dictionary[move] = True
            self.action_dictionary[move][0] += 1
        return self.the_best_move


