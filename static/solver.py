import numpy as np


class Model:
    POSITIVE_FIELD_LENGTH = 11
    NEGATIVE_FIELD_BOX_LENGTH = 1
    OBS_RADIUS = 5
    BONUS_FIELD = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, ],
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, ],
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, 2, 2, 1, 0, 0, ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, ],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, ],
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, 2, 2, 1, 0, 0, ],
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, ],
        [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],

    ])

    def __init__(self):
        self.path = None

    @staticmethod
    def get_target_point(obs: object) -> object:
        """
            Функция получения координат цели или проекции цели
        """
        pos_i, pos_j = np.where(obs == 1)
        return pos_i[0], pos_j[0]

    @staticmethod
    def target_in_visibility(obs):
        """
            Функция определяет: находится ли цель в области видимости агента
        """
        return 1 in obs[1:-1, 1:-1]

    @staticmethod
    def get_positive_field(obs):
        """
          Функция создает положительное потенциальное поле вокруг цели
        """
        pos_i, pos_j = Model.get_target_point(obs)
        positive_field = Model.BONUS_FIELD[10 - pos_i:21 - pos_i, 10 - pos_j:21 - pos_j]
        return positive_field

    @staticmethod
    def get_negative_field_box(obs):
        """
            Функция создает отрицательное потенциальное поле вокруг боксов
        """
        return -300 * obs

    @staticmethod
    def preprocess_boxes(obs, target):
        """
            Функция избавляется от тупиковых точек
        """

        def is_dead_block(i, j, current_obs):
            """
                Функция определяет: является ли переданная точка тупиком
            """
            current_obs = np.pad(current_obs, 1)
            return (current_obs[i, j + 1] + current_obs[i + 2, j + 1] + current_obs[i + 1, j] + current_obs[
                i + 1, j + 2]) >= 3

        result = obs.copy()
        dead_block_find = True
        while dead_block_find:
            dead_block_find = False
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if (i, j) == target or (i, j) == (5, 5):
                        continue
                    if result[i, j] == 0:
                        if is_dead_block(i, j, result):
                            result[i, j] = 1
                            dead_block_find = True
        return result

    @staticmethod
    def out_of_range(point):
        """
            Функция проверяет выход точки за пределы поля
        """
        return point[0] < 0 or point[0] > Model.OBS_RADIUS * 2 or point[1] < 0 or point[1] > Model.OBS_RADIUS * 2

    def get_action(self, idx, obs):
        """
            Функция определяет действие
        """
        target = self.__class__.get_target_point(obs[2])
        pos = self.__class__.get_positive_field(obs[2])
        new_obs = self.__class__.preprocess_boxes(obs[0], target)
        new_neg = self.__class__.get_negative_field_box(new_obs)
        enemy_neg = self.__class__.get_negative_field_box(obs[1])
        result = pos + new_neg + enemy_neg

        steps = {
            0: (0, 0),
            1: (1, 0),  # up
            2: (-1, 0),  # down
            3: (0, 1),  # left
            4: (0, -1),  # right
        }
        current_point = np.array([5, 5])
        result[tuple(current_point)] = -256
        for i, p in enumerate(self.path[idx][::-1]):
            current_point = np.sum([current_point, steps[p]], axis=0)
            if not self.__class__.out_of_range(current_point):
                result[tuple(current_point)] = np.min([-256 + i, result[tuple(current_point)]])

        action = np.argmax([
            -258,
            result[4, 5],  # наверх
            result[6, 5],  # вниз
            result[5, 4],  # влево
            result[5, 6], ]  # вправо
        )
        # print(result)
        # f, ax = plt.subplots(1,2, figsize = (8,6))
        # ax[0].imshow(result, cmap='hot')
        # plt.show()
        # print(action)

        self.path[idx].append(action)
        return action

    def act(self, obs, done, positions_xy, targets_xy):
        if self.path is None:
            self.path = [[0] for _ in range(len(obs))]
        return [self.get_action(i, o) for i, o in enumerate(obs)]
