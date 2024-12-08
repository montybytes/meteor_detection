from itertools import chain


def mapFromTo(x, a, b, c, d):
    """Routine to map values from one range to another

    Args:
        x: value to map
        a: source minimum range value
        b: source maximum range value
        c: target minimum range value
        d: target maximum range value
    """

    return c + (x - a) * (d - c) / (b - a)


def flatten(array):
    """Routine that takes n-dimensional arrays and converts them to 1-dimensional"""

    return list(dict.fromkeys(list(chain.from_iterable(array))))
