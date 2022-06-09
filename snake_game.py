import random
import pygame
from painter import Painter
import time


class SnakeGame:
    def __init__(self, size_x, size_y, line_size=1, speed=10):
        self.speed = speed
        self.food_pos = [0, 0]
        self.box_size = 40
        self.line_size = line_size
        self.painter = Painter(size_x * (self.box_size + self.line_size),
                               size_y * (self.box_size + self.line_size))
        self.board = []
        self.size_x = size_x
        self.size_y = size_y
        self.list_snake_body = []
        self.create_board()
        self.direction = (1, 0)
        self.score = 0
        self.dead = False
        self.reward = 0

    def reset(self):
        self.food_pos = [0, 0]
        self.board = []
        self.list_snake_body = []
        self.create_board()
        self.direction = (1, 0)
        self.score = 0
        self.dead = False
        return [self.board]

    def create_board(self):
        self.board = [[0 for y in range(self.size_y)] for x in range(self.size_x)]
        center_x = self.size_x // 2
        center_y = self.size_y // 2
        head_pos = [center_x + 1, center_y]
        body_pos = [center_x, center_y]
        body_pos_2 = [center_x - 1, center_y]
        body_pos_3 = [center_x - 2, center_y]
        self.board[center_x][center_y] = 1
        self.board[center_x + 1][center_y] = 2
        self.list_snake_body = []
        self.list_snake_body.append(body_pos)
        self.list_snake_body.append(body_pos_2)
        self.list_snake_body.append(body_pos_3)
        self.head_pos = head_pos
        self.direction = [1, 0]
        self.create_food()

    def game_logic(self):
        last_direction = self.direction
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    if last_direction != (0, 1):
                        self.direction = (0, -1)
                elif event.key == pygame.K_s:
                    if last_direction != (0, -1):
                        self.direction = (0, 1)
                elif event.key == pygame.K_a:
                    if last_direction != (1, 0):
                        self.direction = (-1, 0)
                elif event.key == pygame.K_d:
                    if last_direction != (-1, 0):
                        self.direction = (1, 0)
                if event.key == pygame.K_ESCAPE:
                    exit()
        self.make_move()
        self.is_death()
        self.eat()

    def make_move(self):
        last_pos = self.head_pos.copy()
        last_pos_2 = [0, 1]
        tail_pos = self.list_snake_body[-1].copy()
        self.board[tail_pos[0]][tail_pos[-1]] = 0
        self.head_pos[0] = (self.head_pos[0] + self.direction[0]) % self.size_x
        self.head_pos[1] = (self.head_pos[1] + self.direction[1]) % self.size_y
        self.board[self.head_pos[0]][self.head_pos[1]] = 2

        for snake_body in self.list_snake_body:
            last_pos_2[0] = snake_body[0]
            last_pos_2[1] = snake_body[1]
            snake_body[0] = last_pos[0]
            snake_body[1] = last_pos[1]
            last_pos[0] = last_pos_2[0]
            last_pos[1] = last_pos_2[1]
        for snake_body in self.list_snake_body:
            self.board[snake_body[0]][snake_body[1]] = 1

    def game_loop(self):
        while not self.dead:
            time.sleep(1 / self.speed)
            self.game_logic()
            self.draw_board()

    def draw_board(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.board[x][y] == 0:
                    self.painter.draw_empty_box(x * (self.box_size + self.line_size),
                                                y * (self.box_size + self.line_size), self.box_size)
                elif self.board[x][y] == 1:
                    self.painter.draw_snake_box(x * (self.box_size + self.line_size),
                                                y * (self.box_size + self.line_size), self.box_size)
                elif self.board[x][y] == 2:
                    self.painter.draw_snake_head_box(x * (self.box_size + self.line_size),
                                                     y * (self.box_size + self.line_size),
                                                     self.box_size)
                elif self.board[x][y] == 3:
                    self.painter.draw_food_box(x * (self.box_size + self.line_size),
                                               y * (self.box_size + self.line_size),
                                               self.box_size)
        self.painter.update()

    def draw_path(self, path):
        for box in path:
            x, y = box
            self.painter.draw_path(x * (self.box_size + self.line_size), y * (self.box_size + self.line_size),
                                   self.box_size)
        self.painter.update()

    def is_death(self):
        if self.board[self.head_pos[0]][self.head_pos[1]] == 1:
            self.dead = True
            self.score -= 2

    def eat(self):
        if self.head_pos == self.food_pos:
            self.score += 1
            # print("zjedzone", "pkt=", self.score)
            self.list_snake_body.append(self.food_pos.copy())
            self.create_food()

    def create_food(self):
        food_x = random.randint(0, self.size_x - 1)
        food_y = random.randint(0, self.size_y - 1)
        if self.board[food_x][food_y] == 0:
            self.board[food_x][food_y] = 3
            self.food_pos = [food_x, food_y]
        else:
            self.create_food()

    def make_step(self, ai_move):
        start_distance = self.distance_to_food()
        last_score = self.score
        last_direction = self.direction
        # print(ai_move)
        if ai_move == 0:
            ai_move = (1, 0)
        elif ai_move == 1:
            ai_move = (-1, 0)
        elif ai_move == 2:
            ai_move = (0, 1)
        elif ai_move == 3:
            ai_move = (0, -1)
        if last_direction[0] != -ai_move[0] and last_direction[1] != -ai_move[1]:
            self.direction = ai_move
        self.make_move()
        self.is_death()
        end_distance = self.distance_to_food()
        self.eat()
        reward_distance = start_distance - end_distance
        reward = self.score - last_score
        reward += reward_distance * 0.125

        return [self.board], reward, self.dead

    def get_board(self):
        return self.board

    def possible_move(self):
        return [[0, 1], [1, 0], [-1, 0], [0, -1]]

    def distance_to_food(self):
        return abs(self.food_pos[0] - self.head_pos[0]) + abs(
            self.food_pos[1] - self.head_pos[1])

    def get_simple_state(self):
        state = []
        # food position state
        if self.food_pos[0] < self.head_pos[0]:
            state.append(-1)
        elif self.food_pos[0] > self.head_pos[0]:
            state.append(1)
        else:
            state.append(0)
        if self.food_pos[1] < self.head_pos[1]:
            state.append(-1)
        elif self.food_pos[1] > self.head_pos[1]:
            state.append(1)
        else:
            state.append(0)
        direction_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for direction in direction_list:
            if self.is_empty(direction):
                state.append(1)
            else:
                state.append(-1)
        for i, direction in enumerate(direction_list):
            if direction == self.direction:
                state.append(i)
        state = tuple(state)
        return state

    def is_empty(self, direction):
        head_x, head_y = self.head_pos
        next_x = (head_x + direction[0]) % self.size_x
        next_y = (head_y + direction[1]) % self.size_y
        if self.board[next_x][next_y] == 1:
            return False
        return True
