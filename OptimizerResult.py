
class OptimizerResult:

    '''
    Класс для хранения и обработки результатов оптимизации
    '''

    def __init__(self, x_opt, f_opt, f_evals, f_eval_errs, status, status_message, bounds=None, constraints=None):
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.f_evals = f_evals
        self.f_eval_errs = f_eval_errs
        self.status = status
        self.status_message = status_message
        if bounds:
            self.bounds_summary = ConstraintsSummary(bounds)
        if constraints:
            self.constraints_summary = ConstraintsSummary(constraints)

    def __str__(self):
        text_string = ''
        text_string += f'x_opt -- {self.x_opt}\n'
        text_string += f'f_opt -- {self.f_opt}\n'
        text_string += f'f_evals -- {self.f_evals}\n'
        text_string += f'f_evals_errs -- {self.f_evals_errs}\n'
        text_string += f'status -- {self.status}\n'
        text_string += f'message -- {self.status_message}\n'

        if hasattr(self, 'bounds_summary'):
            text_string += 'bounds\n'
            text_string += str(self.bounds_summary)

        if hasattr(self, 'constraints_summary'):
            text_string += 'constraints\n'
            text_string += str(self.constraints_summary)

        return text_string


class ConstraintsSummary(dict):
    '''
    Класс для обработки результатов проверки границ варьируемых параметров или функциональных ограничений
    1-го и 2-го рода после завершения процесса оптимизации
    '''
    def __init__(self, constraints):
        for num, constraint in enumerate(constraints):
            if constraint.name:
                self[constraint.name] = constraint.errors
            else:
                self[f'{num}'] = constraint.errors

    def __str__(self):
        text_string = ''
        for key, value in self.items():
            text_string += f'\t{key} ограничение -- {value} ошибок\n'
        return text_string

if __name__ == '__main__':
    from Constraints import Bounds, Constraint
    bound = Bounds(1, 2)
    cons = Constraint(lambda x: x, 1, 2)
    con2 = Constraint(lambda x: x, 1, 2)

    #cons_sumry = ConstraintsSummary((cons, con2))

    res = OptimizerResult(1, 1, 5, True, 'Успешно', bounds=(bound, ), constraints=(cons, con2))
    print(res)