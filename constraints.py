
class Constraint:

    def __init__(self, func, lbound, ubound, name=''):
        self.func = func
        self._lbound = lbound
        self._ubound = ubound
        self._name = name
        self._errors = 0

    def check(self, x, *args):
        # Проверка границ(для не сеточных методов)
        res = self._lbound <= self.func(x, *args) <= self._ubound
        if not res:
            self._errors += 1
        return res

    @property
    def name(self):
        return self._name

    @property
    def lbound(self):
        return self._lbound

    @property
    def ubound(self):
        return self._ubound

    @property
    def errors(self):
        return self._errors

    def clear(self):
        self._errors = 0


class Bounds(Constraint):

    def __init__(self, lbound, ubound, name=''):

        super().__init__(func=lambda x, *args: x, lbound=lbound, ubound=ubound, name=name)

    @classmethod
    def from_tuple(cls, tup):
        return cls(tup[0], tup[1])

    def to_list(self):
        return [self.lbound, self.ubound]


if __name__ == '__main__':
    bound = Bounds(1, 2)
    cons = Constraint(lambda x: x, 1, 2)