class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith('__'))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()


class TrainMethod(metaclass=ContainerMeta):
    DARE = 'DARE'
    ITERATIVE = 'Analytical Iterative'
    GPI = 'Generalized Policy Iteration'


class PolicyEvaluation(metaclass=ContainerMeta):
    QLEARN = 'Q-learning'
    QLEARN_RLS = 'RLS Q-learning'


class PolicyType(metaclass=ContainerMeta):
    RANDOM = 'random'
    EPS_GREEDY = 'epsilon greedy'
    GREEDY = 'greedy'


class NoiseShape(metaclass=ContainerMeta):
    MV_NORMAL = 'MV Normal'


class ConvergenceCriteria(metaclass=ContainerMeta):
    K_INF_NORM = 'K Inf Norm'
