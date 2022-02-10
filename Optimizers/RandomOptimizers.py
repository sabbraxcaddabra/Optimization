from abc import ABC, abstractmethod
import numpy as np
from numpy.random import uniform
from numpy.linalg import norm

from Optimization.OptimizationErrors import *

# TODO: Добавить методы для удаления ограничений 1-го и 2-го рода по ключу
# TODO: Проверить качество и правильность реализации метода случайного сканирования
# TODO: Добавить еще несколько реализацций алгоритмов оптимизации
# TODO: Написать нормальную документацию для каждого класса



class Optimizer(ABC):
    """
    Абстрактный класс-оптимайзер.
    Конкретные реализации алгоритмов оптимизации должны наследоваться от данного класса
    """
    possible_errors = [LimitExceedError, FirstGroundBoundaryError, SecondGroundBoundaryError]
    def __init__(self,
                 x_vec,
                 params=None,
                 adapters=dict(),
                 first_ground_boundary=dict(),
                 second_ground_boundary=dict(),
                 x_lims=None,
                 t_func=None,
                 out_func=None):
        self.x_vec = x_vec
        self.params = params
        self.adapters = adapters
        self.first_ground_boundary = first_ground_boundary
        self.second_ground_boundary = second_ground_boundary
        self.x_lims = x_lims
        self.t_func = t_func
        self.out_func = out_func

    def add_new_adapter(self, key, adapt_func) -> None:
        """
        Метод для добавления в задачу нового адаптера
        :param adapt_func: Функция, лямбда-функция, классовый метод и т.д(callable объект)
        :return: None
        """
        self.adapters[key] = adapt_func

    def _adapt(self, x_vec_new: list) -> None:
        """
        Метод адаптирует параметры задачи(подставляет значения x_vec_new в необходимые поля params для решения
        целевой функции
        :param x_vec_new: Новый вектор варьируемых параметров X
        :return: None
        """
        if self.adapters:
            for func in self.adapters.values():
                func(x_vec_new, self.params)

    def remove_adapter(self, key):

        del self.adapters[key]

    def add_first_ground_boundary(self, name: str, func_dict: dict) -> None:
        """
        Метод для добавления функций - ограничений первого рода
        :param name: Название ограничения первого рода(оптимизатором не используется, необходимо для удобства
        пользователя и возможности проще удалить при необходимости)
        :param func_dict: Словарь с ключами func и lims, где func соответсвтует функция, лямбда и тд,
        которая принимат в себе параметры задачи и сравнивает необходимые поля параметров или их преобразования с lims
        :return: None
        """
        self.first_ground_boundary[name] = func_dict

    def _check_first_ground_boundary(self, x_vec_cur):
        """
        Проверка ограничений 1-го рода. Если словарь ограничений пуст, проверяется только вхождение каждой компоненты
        x_vec_cur в ограничения заданные x_lims
        :param x_vec_cur: Текущая реализация вектора варьируемых параметров
        :return: bool
        """
        if self.first_ground_boundary:
            for func_dict_name, func_dict in self.first_ground_boundary.items():
                res = func_dict['func'](x_vec_cur, self.params, func_dict['lim'])
                if not res:
                    func_dict['errors'] += 1
                    raise FirstGroundBoundaryError(func_dict_name)

        if len(self.x_lims) != len(x_vec_cur):
            raise Exception("Длина вектора варьируемых параметров не совпадает с длиной вектора ограничений")
        else:
            check_list = [lim[0] <= x <= lim[1] for lim, x in zip(self.x_lims, x_vec_cur)]
            if not all(check_list):
                raise LimitExceedError(x_vec_cur, self.x_lims)


    def add_second_ground_boundary(self, name: str, func_dict: dict) -> None:
        """
        Добавление функций-ограничений второго рода
        :param name: Название ограничения первого рода(оптимизатором не используется, необходимо
        для удобства пользователяи возможности проще удалить при необходимости)
        :param func_dict: Словарь с ключами func и lims, где func соответсвтует функция,
        лямбда и тд, которая принимат в себея параметры
        задачи и решение целевой функции и сравнивает необходимые поля решения с lims
        :return: None
        """
        self.second_ground_boundary[name] = func_dict

    def _check_second_ground_boundary(self, x, y, solution, params):
        """
        Проверка ограничений 2-го рода
        :param solution: Текущий результат решения целевой функции
        :return: bool
        """

        if self.second_ground_boundary:
            for func_dict_name, func_dict in self.second_ground_boundary.items():
                res = func_dict['func'](y, solution, self.params, func_dict['lim'])
                if not res:
                    fine_func = func_dict.get('fine')
                    if fine_func:
                        y = fine_func(x, y, solution, params)
                        func_dict['fines'] += 1
                    else:
                        func_dict['errors'] += 1
                        raise SecondGroundBoundaryError(func_dict_name)
        return y

    def set_target_func(self, t_func) -> None:
        """
        Установка целевой функции
        :param t_func: Целевая функция(callable)
        :return:
        """
        self.t_func = t_func

    def set_out_func(self, o_func):
        self.out_func = o_func

    def clearify_errors(self):
        '''
        Зануление счетчиков ошибок для ограничений 1 и 2 рода
        :return:
        '''
        if self.second_ground_boundary:
            for func_dict in self.second_ground_boundary.values():
                func_dict['errors'] = 0
                func_dict['fines'] = 0

        if self.first_ground_boundary:
            for func_dict in self.first_ground_boundary.values():
                func_dict['errors'] = 0

    def get_optimization_summary(self, above_limits, target_func_calculated):
        summary = {'t_func_calcs': target_func_calculated, 'first_ground': dict(), 'above_limits': above_limits, 'second_ground': dict()}

        if self.second_ground_boundary:
            for func_key, func_value in self.second_ground_boundary.items():
                summary['second_ground'][func_key] = {'errors': func_value['errors']}
                summary['second_ground'][func_key]['fines'] = func_value['fines']

        if self.first_ground_boundary:
            for func_key, func_value in self.first_ground_boundary.items():
                summary['first_ground'][func_key] = {'errors': func_value['errors']}

        return summary


    @abstractmethod
    def optimize(self):
        pass