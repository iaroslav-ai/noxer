def check_in(a_element, b_iterable, param_name):
    """
    Checks if a_element is in b_iterable. If not,
    the message is shown.

    Parameters:
    -----------

    a_element: object
        object whose presence in some iterable will be
        checked.

    b_iterable: iterable
        iterable where the presence of element is checked.

    param_name:
        Name of the parameter being checked.
    """

    if a_element in b_iterable:
        return

    raise ValueError(
        "%s should be one of %s, but got %s." % (param_name, a_element, b_iterable)
    )

def check_rng(a_element, b_rng, param_name):
    """
    Checks if a_element is in range b_rng. If not,
    the message is shown.

    Parameters:
    -----------

    a_element: object
        object whose presence in some iterable will be
        checked.

    b_rng: iterable of form [a, b]
        feasible interval. Upper and lower bounds included.

    param_name:
        Name of the parameter being checked.
    """

    if b_rng[0] <= a_element <= b_rng[1]:
        return

    raise ValueError(
        "%s should be in the interval %s, but got %s." % (param_name, a_element, b_rng)
    )

def check_true(cond, error_message):
    """
    Checks if cond. If not,
    the message is shown.

    Parameters:
    -----------

    cond: boolean
        condition to be checked.
    """

    if cond:
        return

    raise AssertionError(
        error_message
    )

def check_isinst(example, clazz, param_name):
    """
    Checks if example is an instance of example.
    If not, the message is shown.

    Parameters:
    -----------

    example: object
        object whose type will be checked.

    clazz: iterable
        class type.

    error_message:
        Error message to be shown.
    """

    if isinstance(example, clazz):
        return

    raise ValueError(
        "%s parameter should be of type: %s, but got type: %s" % (param_name, clazz, type(example))
    )