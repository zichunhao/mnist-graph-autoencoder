
"""
Expand a variable to a list of indicated length.
"""
def expand_var_list(var, length):
    if isinstance(var, list):
        if len(var) < length:
            var += [var[-1]] * length
    elif type(var) in [float, int]:
        var = [var] * length
    else:
        raise ValueError(f"Incorrect type of data: {type(var)}. Data must be of type list, float, or int!")
    return var
