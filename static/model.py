import random
import numpy as np


class Model:
    OBS_RADIUS = 5
    RADIUS = 21
    BONUS_FIELD = None
    MASK = np.zeros((11, 11))
    steps = {
        0: (0, 0),
        1: (1, 0),  # up
        2: (-1, 0),  # down
        3: (0, 1),  # left
        4: (0, -1),  # right
    }

    def __init__(self):
        self.__class__.MASK[0] = 1
        self.__class__.MASK[1] = 1
        self.__class__.MASK[-1] = 1
        self.__class__.MASK[-2] = 1
        self.__class__.MASK[:, -1] = 1
        self.__class__.MASK[:, -2] = 1
        self.__class__.MASK[:, 0] = 1
        self.__class__.MASK[:, 1] = 1
        self.path = None

    @staticmethod
    def get_target_point(obs):
        """
            Функция получения координат цели или проекции цели
        """
        pos_i, pos_j = np.where(obs == 1)
        return (pos_i[0], pos_j[0])

    @staticmethod
    def target_in_visibility(obs):
        """
            Функиця определяет: находится ли цель в области видимости агента
        """
        return 1 in obs[1:-1, 1:-1]

    @staticmethod
    def get_positive_field(obs):
        """
            Функция создает положительное потенциальное поле вокруг цели
        """
        pos_i, pos_j = Model.get_target_point(obs)
        positive_field = Model.BONUS_FIELD[
                         Model.RADIUS - pos_i:Model.RADIUS + 11 - pos_i,
                         Model.RADIUS - pos_j:Model.RADIUS + 11 - pos_j]
        return positive_field

    @staticmethod
    def get_negative_field_box(obs):
        """
            Функция создает отрицательное потенциальное поле вокруг боксов
        """
        return -400 * obs

    @staticmethod
    def get_enemy_field(obs):
        """
            Функция создает небольшое отрицательное поле вокруг других агентов
        """
        enemy_negative_field = np.zeros_like(obs)
        points = np.where(obs == 1)
        for i, j in zip(*points):
            if (i, j) == (5, 5):
                continue
            # enemy_negative_field[
            #                      np.clip(0, obs.shape[0], i - 2):np.clip(0, obs.shape[0], i + 3),
            #                      np.clip(0, obs.shape[1], j - 2):np.clip(0, obs.shape[0], j + 3), ] += 0.33

            # enemy_negative_field[np.clip(0, obs.shape[0], i - 1):np.clip(0, obs.shape[0], i + 2),
            # np.clip(0, obs.shape[1], j - 1):np.clip(0, obs.shape[0], j + 2), ] += 0.33
            enemy_negative_field[i, j] += 1
        return enemy_negative_field

    @staticmethod
    def preprocess_boxes(obs, target):
        """
            Функция избавляется от тупиковых точек
        """

        penalty_value = 0.8

        def is_simple_dead_block(point, result):
            """
                Функция определяет: является ли переданная точка тупиком
            """
            i, j = point
            current_obs = np.pad(result, 1)
            return (current_obs[i, j + 1] + current_obs[i + 2, j + 1] + current_obs[i + 1, j] + current_obs[
                i + 1, j + 2]) >= 3

        result = obs.copy()
        dead_block_find = True
        while dead_block_find:
            dead_block_find = False
            # Получение всех свободных точек
            points = np.argwhere((result == 0) & (Model.MASK == 1))
            # Проход по всем найденным точкам
            for point in points:
                if tuple(point) == target or tuple(point) == (5, 5):
                    continue
                if is_simple_dead_block(point, result):
                    result[tuple(point)] = penalty_value
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
        obs = obs.astype(np.float16)
        target = self.__class__.get_target_point(obs[2])
        pos = self.__class__.get_positive_field(obs[2])
        # new_obs = self.__class__.preprocess_boxes(obs[0], target)
        new_neg = self.__class__.get_negative_field_box(obs[0])
        enemy_neg = self.__class__.get_enemy_field(obs[1])
        enemy_neg = self.__class__.get_negative_field_box(enemy_neg)
        result = pos + new_neg + enemy_neg

        current_point = np.array([5, 5])
        result[tuple(current_point)] = -256
        for i, p in enumerate(self.path[idx][::-1]):
            current_point = np.sum([current_point, self.__class__.steps[p]], axis=0)
            if not self.__class__.out_of_range(current_point):
                result[tuple(current_point)] = np.min([-256 + i, result[tuple(current_point)]])
        action = np.argmax([
            -258,
            result[4, 5],  # наверх
            result[6, 5],  # вниз
            result[5, 4],  # влево
            result[5, 6], ]  # вправо
        )

        self.path[idx].append(action)
        return action

    def act(self, obs, done, positions_xy, targets_xy):
        if self.path is None:
            self.path = [[0] for _ in range(len(obs))]
            Model.BONUS_FIELD = np.zeros((Model.RADIUS * 2 + 1, Model.RADIUS * 2 + 1))
            for i in range(Model.BONUS_FIELD.shape[0]):
                for j in range(Model.BONUS_FIELD.shape[1]):
                    Model.BONUS_FIELD[i, j] = np.clip(0, Model.RADIUS,
                                                      ((Model.RADIUS - i) ** 2 + (Model.RADIUS - j) ** 2) ** 0.5)
            Model.BONUS_FIELD = np.max(Model.BONUS_FIELD) - Model.BONUS_FIELD
        return [self.get_action(i, o) for i, o in enumerate(obs)]
