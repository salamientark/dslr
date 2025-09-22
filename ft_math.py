import pandas as pd


def ft_isnbr(elem) -> bool:
    """Check if an element is a number (int or float).

    Parameters:
      elem: Element to check.

    Returns:
      bool: True if the element is a number, False otherwise.
    """
    if isinstance(elem, (int, float)) and not pd.isna(elem):
        return True
    return False


def ft_mean(
        array,
        count: int | None = None
        ) -> float:
    """Compute the mean of a list of numbers.

    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                              will be computed.

    Returns:
      float: Mean of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    mean = 0
    for i in range(c):
        mean += (1. / float(c)) * array[i]
    return mean


def ft_variance(
        array,
        mean: float | None = None,
        count: int | None = None
        ) -> float:
    """Compute the variance of a list of numbers.

    Parameters:
      array (list): List of numbers.
      mean (float) (optional): Mean of the list of numbers. If None, it will
                               be computed.
      count (int) (optionnal): Number of elements in the list. If None, it
                               will be computed.

    Returns:
      float: Variance of the list of numbers.
    """
    c = count
    m = mean
    if m is None:
        m = ft_mean(array)
    if c is None:
        c = len(array)
    var = 0
    for i in range(c):
        var += (1. / float(c)) * (array[i] - m) ** 2
    return var


def ft_std(
        array,
        var: float | None = None,
        count: int | None = None
        ) -> float:
    """Compute the standard deviation of a list of numbers.

    Parameters:
      array (list): List of numbers.
      var (float) (optional): Variance of the list of numbers. If None,
                              it will be computed.
      count (int) (optionnal): Number of elements in the list. If None, it
                               will be computed.
    Returns:
      float: Standard deviation of the list of numbers.
    """
    c = count
    v = var
    if c is None:
        c = len(array)
    if v is None:
        v = ft_variance(array, count=c)
    return v ** 0.5


def ft_min(array) -> int | float:
    """Return the minimum of a list of numbers.

    Parameters:
      array (list): List of numbers.

    Returns:
      int | float: Minimum of the list of numbers.
    """
    sorted_array = sorted(array)
    return sorted_array[0]


def ft_max(array) -> int | float:
    """Return the maximum of a list of numbers.

    Parameters:
      array (list): List of numbers.

    Returns:
      int | float: Maximum of the list of numbers.
    """
    sorted_array = sorted(array)
    return sorted_array[-1]


def ft_q1(array, count: int | None = None) -> int | float:
    """Return the first quartile of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: First quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[c // 4]


def ft_q2(array, count: int | None = None) -> int | float:
    """Return the second quartile (median) of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Second quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[c // 2]


def ft_q3(array, count: int | None = None) -> int | float:
    """Return the third quartile of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Third quartile of the list of numbers.
    """
    c = count
    if c is None:
        c = len(array)
    sorted_array = sorted(array)
    return sorted_array[(c // 4) * 3]
