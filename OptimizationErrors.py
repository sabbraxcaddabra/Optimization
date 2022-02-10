class TooMuchItersOptimizerError(Exception):

    def __init__(self, optimization_summary: dict):
        self.optimization_summary = optimization_summary
        super().__init__()

    def __str__(self):
        return "Не найдено ни одного оптимума. Достигнут минимальный шаг\n" \
               "Попробуйте меньший шаг или большее максимальное число итераций" + '\n' + str(self.optimization_summary)


class MinStepOptimizerError(Exception):

    def __init__(self, optimization_summary: dict):
        self.optimization_summary = optimization_summary
        super().__init__()

    def __str__(self):
        return "Не найдено ни одного оптимума. Достигнут минимальный шаг\n" \
               "Попробуйте меньший шаг или большее максимальное число итераций" + '\n' + str(self.optimization_summary)


class SecondGroundBoundaryError(Exception):

    def __init__(self, name):
        self.name = name
        super().__init__()

    def __str__(self):
        return f'Ограничение второго рода {self.name} не пройдено.'


class FirstGroundBoundaryError(Exception):

    def __init__(self, name):
        self.name = name
        super().__init__()

    def __str__(self):
        return f'Ограничение первого рода {self.name} не пройдено.'


class LimitExceedError(Exception):

    def __init__(self, x_vec, x_lims):
        self.x_vec = list(x_vec)
        self.x_lims = list(x_lims)

    def __str__(self):
        return f'Вектор варьируемых параметров за границей поиска.'


class UnknownError(Exception):

    def __str__(self):
        return "Неизвестная ошибка в целевой функции."


class FirstStepOptimizationFail(Exception):

    def __init__(self, error=UnknownError(), t_func_info=None, out_message=None,
                 message="Ошибка при первой попытке вычисления целевой функции.\n"):
        self.error = error
        self.t_func_info = t_func_info
        if isinstance(out_message, str):
            self.out_message = out_message
        else:
            self.out_message = 'Целевая функция не вычислялась'
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message + str(self.error) + '\n\n' + 'Детали расчета:' + \
               '\n' + self.out_message

