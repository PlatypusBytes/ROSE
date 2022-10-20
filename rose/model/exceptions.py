"""
This module contains a set of Exception classes, which give the user more clarity, on what went wrong during calculation

"""

class ParameterNotDefinedException(Exception):
    """
    Exception which indicates that a parameter is not defined. This class bases from :class:`Exception`

    """
    pass


class TimeException(Exception):
    """
    Exception which indicates that there is a problem in the time discretisation. This class bases
    from :class:`Exception`

    """
    pass


class SizeException(Exception):
    """
    Exception which indicates that an element has an incorrect size

    """
    pass