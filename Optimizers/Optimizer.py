from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Абстрактный класс-оптимайзер.
    Конкретные реализации алгоритмов оптимизации должны наследоваться от данного класса
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def show_options(self):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    def _check_bounds(self, x_vec_cur, bounds):
        if bounds:
            for x, bound in zip(x_vec_cur, bounds):
                res = bound.check(x)
                if not res:
                    return res
        return True

    def _check_constraints(self, x_vec_cur, constraints, args):
        if constraints:
            for constraint in constraints:
                res = constraint.check(x_vec_cur, *args)
                if not res:
                    return res
        return True