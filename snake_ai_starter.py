from snake_game import SnakeGame
from ai import AI
import time


def start_ai_game():
    snake_game = SnakeGame(15, 15, line_size=1, speed=100)
    ai = AI(snake_game)
    while True:
        time.sleep(0.05)
        path = ai.make_move()
        snake_game.draw_board()
        snake_game.draw_path(path[1:])
        if snake_game.dead:
            snake_game.reset()
            ai.reset()


start_ai_game()
