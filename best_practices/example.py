import functools
import logging

logging.getLogger().setLevel(logging.INFO)

def add(*args: int) -> int:
    """Add numbers

    Returns:
        int: numbers to add
    """
    assert all([isinstance(arg, int) for arg in args])
    logging.info(args)
    return functools.reduce(lambda a, b: a + b, args, 0)
