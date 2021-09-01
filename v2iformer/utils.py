
# helper functions

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth
