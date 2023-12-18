from ._parameters import Parameter


class ScalarParameter(Parameter):
    pass


class UniformParameter(ScalarParameter):
    pass


class GaussianParameter(ScalarParameter):
    pass


class CustomParameter(ScalarParameter):
    pass
