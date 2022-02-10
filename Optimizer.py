from abc import ABC, abstractmethod
from OptimizationErrors import *
from Constraints import *

class Optimizer(ABC):
    """
    Абстрактный класс-оптимайзер.
    Конкретные реализации алгоритмов оптимизации должны наследоваться от данного класса
    """
    possible_errors = [LimitExceedError, FirstGroundBoundaryError, SecondGroundBoundaryError]

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def show_options(self):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    def _check_constraints(self, x_vec_cur, constraints):
        if constraints:
            for x, constraint in zip(x_vec_cur, constraints):
                res = constraint.check(x)
                if not res:
                    return res
        return True