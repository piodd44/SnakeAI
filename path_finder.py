import time

import numpy as np
import copy


class PathFinder:
    def __init__(self):
        self.block = -1
        self.good_way = -2
        self.no_way = 222222

    def findPath(self, start, end, board,obstacle):
        start_x, start_y = start
        end_x, end_y = end
        temp_arr = copy.deepcopy(board)
        temp_arr[end_x][end_y] = -3
        my_queue = [start]
        for x in range(len(temp_arr)):
            for y in range(len(temp_arr[x])):
                if temp_arr[x][y] in obstacle:
                    temp_arr[x][y] = self.block
                else:
                    temp_arr[x][y] = self.good_way
        temp_arr[end_x][end_y] = -3
        temp_arr[start_x][start_y] = 0

        while len(my_queue) > 0:
            cur_start = my_queue.pop()
            self.check_one_point(start=cur_start, direction=(1, 0), temp_arr=temp_arr, my_queue=my_queue)
            self.check_one_point(start=cur_start, direction=(-1, 0), temp_arr=temp_arr, my_queue=my_queue)
            self.check_one_point(start=cur_start, direction=(0, 1), temp_arr=temp_arr, my_queue=my_queue)
            self.check_one_point(start=cur_start, direction=(0, -1), temp_arr=temp_arr, my_queue=my_queue)

        return self.give_path(end=end, temp_arr=temp_arr)

    def check_one_point(self, start, direction, temp_arr, my_queue):
        cur_x, cur_y = start
        next_x = (cur_x + direction[0]) % len(temp_arr)
        next_y = (cur_y + direction[1]) % len(temp_arr[0])
        if temp_arr[next_x][next_y] == -3:
            return
        elif temp_arr[next_x][next_y] == self.good_way:
            temp_arr[next_x][next_y] = temp_arr[cur_x][cur_y] + 1
            my_queue.append((next_x, next_y))
            return
        elif temp_arr[next_x][next_y] == self.block:
            return
        elif temp_arr[next_x][next_y] > temp_arr[cur_x][cur_y] + 1:
            temp_arr[next_x][next_y] = temp_arr[cur_x][cur_y] + 1
            my_queue.append((next_x, next_y))
            return

    def give_path(self, end, temp_arr):
        path = [end]
        cur = end
        direction = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        distance = -10
        while distance != 0:
            temp = []
            temp_0 = self.one_point_value(cur, direction=(1, 0), temp_arr=temp_arr)
            temp_1 = self.one_point_value(cur, direction=(-1, 0), temp_arr=temp_arr)
            temp_2 = self.one_point_value(cur, direction=(0, 1), temp_arr=temp_arr)
            temp_3 = self.one_point_value(cur, direction=(0, -1), temp_arr=temp_arr)
            temp.append(temp_0)
            temp.append(temp_1)
            temp.append(temp_2)
            temp.append(temp_3)
            k = np.argmin(temp)
            cur = (cur[0] + direction[k][0], cur[1] + direction[k][1])
            path.append(cur)
            distance = temp[k]
            if distance == self.no_way:
                return []
        return path[:-1]

    def one_point_value(self, start, direction, temp_arr):
        cur_x, cur_y = start
        next_x = (cur_x + direction[0]) % len(temp_arr)
        next_y = (cur_y + direction[1]) % len(temp_arr[0])
        if temp_arr[next_x][next_y] < 0:
            return self.no_way
        return temp_arr[next_x][next_y]
