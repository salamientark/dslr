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
    if count is None:
        count = len(array)
    mean = 0
    for i in range(count):
        mean += (1 / count) * array[i]
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
    if mean is None:
        mean = ft_mean(array)
    if count is None:
        count = len(array)
    var = 0
    for i in range(count):
        var += (1 / count) * (array[i] - mean) ** 2
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
    if count is None:
        count = len(array)
    if var is None:
        var = ft_variance(array, count=count)
    return var ** 0.5
    

def ft_min(array) ->int | float:
    """Return the minimum of a list of numbers.

    Parameters:
      array (list): List of numbers.
    
    Returns:
      int | float: Minimum of the list of numbers.
    """
    sorted_array = sorted(array)
    return sorted_array[0]


def ft_max(array) ->int | float:
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
    if count is None:
        count = len(array)
    sorted_array = sorted(array)
    return sorted_array[count // 4]


def ft_q2(array, count: int | None = None) -> int | float:
    """Return the second quartile (median) of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Second quartile of the list of numbers.
    """
    if count is None:
        count = len(array)
    sorted_array = sorted(array)
    return sorted_array[count // 2]


def ft_q3(array, count: int | None = None) -> int | float:
    """Return the third quartile of a list of numbers.
    Parameters:
      array (list): List of numbers.
      count (int) (optional): Number of elements in the list. If None, it
                            will be computed.

    Returns:
      int | float: Third quartile of the list of numbers.
    """
    if count is None:
        count = len(array)
    sorted_array = sorted(array)
    return sorted_array[(count // 4) * 3]
