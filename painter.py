import pygame

pygame.init()

red = [255, 0, 0]
grey = [50, 50, 50]
green = [0, 155, 0]
yellow = [215, 215, 0]
path = [255, 255, 255]


class Painter:
    def __init__(self, size_x, size_y):
        self.screen = pygame.display.set_mode([size_x, size_y])
        self.screen.fill((200, 200, 200))

    def draw_square(self, pos_x, pos_y, size, colour):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=colour, rect=rect)

    def draw_empty_box(self, pos_x, pos_y, size):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=grey, rect=rect)

    def update(self):
        pygame.display.update()

    def draw_snake_box(self, pos_x, pos_y, size):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=red, rect=rect)

    def draw_food_box(self, pos_x, pos_y, size):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=green, rect=rect)

    def draw_snake_head_box(self, pos_x, pos_y, size):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=yellow, rect=rect)

    def draw_path(self, pos_x, pos_y, size):
        rect = pygame.Rect(pos_x, pos_y, size, size)
        pygame.draw.rect(self.screen, color=path, rect=rect)


def test():
    painter = Painter()
    while True:
        painter.draw_square(10, 10, 50, red)

# test()
