from datetime import datetime
from math import isnan


def days_between(start_date, end_date):
    """
    Calculates number of days between two dates in format dd.mm.yyyy.
    If some argument is not a string, returns NaN.
    """
    if not (isinstance(start_date, str) and isinstance(end_date, str)):
        return float("nan")

    start_date = list(map(int, start_date.split(".")[:-1]))
    start_date.reverse()
    end_date = list(map(int, end_date.split(".")[:-1]))
    end_date.reverse()

    start = datetime(*start_date)
    end = datetime(*end_date)

    delta = end - start
    return delta.days
