from path_finder import PathFinder


# from snake_game import SnakeGame


class AI:
    def __init__(self, game):
        self.game = game
        self.path_finder = PathFinder()
        self.path = []

    def reset(self):
        self.path = []

    def make_move(self):
        if len(self.path) == 0:
            self.path = self.path_finder.findPath(start=self.game.head_pos, end=self.game.food_pos,
                                                  board=self.game.board,
                                                  empty=[0], obstacle=[1, 2])
        if len(self.path) == 0:
            move = (1, 0)
        # move = (self.path[-1][0] - self.game.head_pos[0], self.path[-1][1] - self.game.head_pos[1])
        else:
            next_pos = self.path.pop()
            move = (next_pos[0] - self.game.head_pos[0], next_pos[1] - self.game.head_pos[1])
        self.game.make_step(move)
        return self.path
