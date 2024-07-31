from itertools import chain


def mapFromTo(x, a, b, c, d):
    return c + (x - a) * (d - c) / (b - a)


def flatten(array):
    return list(dict.fromkeys(list(chain.from_iterable(array))))
