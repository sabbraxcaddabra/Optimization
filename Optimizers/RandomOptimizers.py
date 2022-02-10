from Optimizer import Optimizer
from OptimizerResult import OptimizerResult
from numpy.random import uniform
from numpy.linalg import norm
import numpy as np



class RandomSearchOptimizer(Optimizer):

    def __init__(self, N=100, M=10, t0=1., R=0.1, alpha=1.618, beta=0.618, min_felta_f=0.):
        self.N = N
        self.M = M
        self.t0 = t0
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.min_delta_f = min_felta_f


    @staticmethod
    def _get_yj(x_cur, tk):
        """

        :param x_cur:
        :param tk:
        :return:
        """
        ksi = uniform(-1, 1, len(x_cur))
        yj = x_cur + tk * ksi / norm(ksi)
        return yj

    @staticmethod
    def _get_zj(x_cur, alpha, yj):
        """

        :param x_cur:
        :param alpha:
        :param yj:
        :return:
        """
        zj = x_cur + alpha * (yj - x_cur)
        return zj

    def optimize(self,
                 t_func,
                 x0,
                 args=tuple(),
                 bounds=None,
                 constraints=None,
                 out_func=None):

        f_evals = 0
        f_evals_errs = 0
        steps_total = 0
        bad_steps_cur = 1

        try:
            last_x = np.ones_like(x0)
            bound_check = self._check_constraints(last_x*x0, bounds)
            last_f = t_func(last_x * self.x_vec, *args)
            f_evals += 1
            constraints_check = self._check_constraints(last_x*x0, bounds)

            if not all((bound_check, constraints_check)):
                return OptimizerResult(
                    last_x*x0,
                    last_f,
                    f_evals=f_evals,
                    f_eval_errs=0,
                    status=False,
                    status_message='Ошибка при проверке ограничений на первом шаге',
                    bounds=bounds,
                    constraints=constraints
                )

        except:
            return OptimizerResult(
                    last_x*x0,
                    last_f,
                    f_evals=1,
                    f_eval_errs=1,
                    status=False,
                    status_message='Ошибка при первом вычислении целевой функции',
                    bounds=bounds,
                    constraints=constraints
                )

        while steps_total < self.N:
            while bad_steps_cur < self.M:

                yj = x0 * self._get_yj(last_x, self.t0)

                try:
                    bounds_check = self._check_constraints(yj, bounds)
                    if bounds_check:
                        cur_f = t_func(yj, *args)
                        f_evals += 1
                        constraints_check = self._check_constraints(yj, constraints)
                        if constraints_check & (cur_f <= last_f) & (abs(cur_f - last_f) > self.min_delta_f):

                            zj = x0 * self._get_zj(last_x, self.alpha, yj/x0)

                            bounds_check = self._check_constraints(zj, bounds)
                            if bounds_check:
                                cur_f = t_func(zj, *args)
                                f_evals += 1
                                constraints_check = self._check_constraints(yj, constraints)

                                if constraints_check & (cur_f <= last_f) & (abs(cur_f - last_f) > self.min_delta_f):
                                    last_x, last_f = zj / x0, cur_f
                                    self.t0 *= self.alpha
                                    steps_total += 1

                                    if self.out_func:
                                        self.out_func(zj)
                                    break
                                else:
                                    bad_steps_cur += 1
                            else:
                                bad_steps_cur += 1
                        else:
                            bad_steps_cur += 1
                    else:
                        bad_steps_cur
                except:
                    bad_steps_cur += 1
                    f_evals += 1
                    f_evals_errs += 1

            if self.t0 <= self.R:

                if not np.array_equal(last_x, np.ones_like(last_x)):
                    return OptimizerResult(
                        last_x*x0,
                        last_f,
                        f_evals=f_evals,
                        f_eval_errs=f_evals_errs,
                        status=False,
                        status_message='Оптимизация завершилась неудачно, достигнут минимальный шаг',
                        bounds=bounds,
                        constraints=constraints
                    )
                else:
                    return OptimizerResult(
                        last_x * x0,
                        last_f,
                        f_evals=f_evals,
                        f_eval_errs=f_evals_errs,
                        status=True,
                        status_message='Оптимизация завершилась удачно, достигнут минимальный шаг',
                        bounds=bounds,
                        constraints=constraints
                    )

            return OptimizerResult(
                last_x * x0,
                last_f,
                f_evals=f_evals,
                f_eval_errs=f_evals_errs,
                status=False,
                status_message='Оптимизация завершилась неудачно, решение не сошлось',
                bounds=bounds,
                constraints=constraints
            )


