import numpy as np


class History(object):

    def __init__(self, obstacles):
        self.maps = []
        for obs in obstacles:
            self.maps.append(obs)
        print(len(self.maps))
        self.relative_pos = np.zeros((len(obstacles), 2), dtype=np.int)   # сдвиг агента относительно стартовой точки
        self.curr_pos = np.ones((len(obstacles), 2), dtype=np.int) * 5    # текущее положение агента в map
        self.edges = np.zeros((len(obstacles), 4), dtype=np.int)          # Пределы куда доходили вгенты на своих картах
                                                                          # [max_up, min_down, min_left, max_right]
        self.steps = {0: (0, 0),   # stay
                      1: (-1, 0),  # up
                      2: (1, 0),   # down
                      3: (0, -1),  # left
                      4: (0, 1)}   # right

    def update_history(self, obstacles, actions):
        actions = np.array(list(map(lambda x: self.steps[x], actions)))

        # обновляем позиции относительно старта и текущие позиции на карте
        self.relative_pos += actions
        self.curr_pos += actions

        for bot, obs in enumerate(obstacles):
            max_up, min_down, min_left, max_right = self.edges[bot]

            # проверяеям выход за верхний предел для агента
            if self.relative_pos[bot, 0] > max_up:
                new_line = np.empty((1, self.maps[bot].shape[1]))
                new_line[:] = np.NaN
                self.maps[bot] = np.concatenate((new_line, self.maps[bot]), axis=0)
                max_up = self.relative_pos[bot, 0]
                self.curr_pos[bot, 0] += 1

            # проверяеям выход за нижний предел для агента
            elif self.relative_pos[bot, 0] < min_down:
                new_line = np.empty((1, self.maps[bot].shape[1]))
                new_line[:] = np.NaN
                self.maps[bot] = np.concatenate((self.maps[bot], new_line), axis=0)
                min_down = self.relative_pos[bot, 0]

            # проверяеям выход за левый предел для агента
            if self.relative_pos[bot, 1] < min_left:
                new_col = np.empty((self.maps[bot].shape[0], 1))
                new_col[:] = np.NaN
                self.maps[bot] = np.concatenate((new_col, self.maps[bot]), axis=1)
                min_left = self.relative_pos[bot, 1]
                self.curr_pos[bot, 1] += 1

            # проверяеям выход за правый предел для агента
            elif self.relative_pos[bot, 1] > max_right:
                new_col = np.empty((self.maps[bot].shape[0], 1))
                new_col[:] = np.NaN
                self.maps[bot] = np.concatenate((self.maps[bot], new_col), axis=1)
                max_right = self.relative_pos[bot, 1]

            # обновляем пределы для агента
            self.edges[bot] = np.array((max_up, min_down, min_left, max_right))

            # обновляем карту агента
            x, y = self.curr_pos[bot]
            self.maps[bot][x-5:x+6, y-5:y+6] = obs

        return self.maps


if __name__ == '__main__':
    obstacles1 = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                           [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                           [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                           [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
                           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                           [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                           [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                           [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]], dtype=np.float)
    obstacles2 = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
                           [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                           [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                           [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                           [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                           [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                           [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=np.float)
    res = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
                    [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=np.float)

    h = History([obstacles1])
    res_h = h.update_history([obstacles2], [4])
    assert np.array_equal(res, res_h[0])
