from snake_game import SnakeGame


def start_human_game():
    snake_game = SnakeGame(size_x=15, size_y=15, line_size=1, speed=10)
    snake_game.game_loop()


start_human_game()
