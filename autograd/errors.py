import sys
import re
from functools import wraps
from future.utils import raise_from, raise_

class AutogradHint(Exception):
    def __init__(self, message, subexception_type=None, subexception_val=None):
        self.message = message
        self.subexception_type = subexception_type
        self.subexception_val = subexception_val

    def __str__(self):
        if self.subexception_type:
            return '{message}\nSub-exception:\n{name}: {str}'.format(
                message=self.message,
                name=self.subexception_type.__name__,
                str=self.subexception_type(self.subexception_val))
        else:
            return self.message

common_errors = [
    ((TypeError, r'float() argument must be a string or a number'),
        "This error *might* be caused by assigning into arrays, which autograd doesn't support."),
    ((ValueError, r'setting an array element with a sequence'),
        "This error *might* be caused by assigning into arrays, which autograd doesn't support."),
    ((TypeError, r"got an unexpected keyword argument '(?:dtype)|(?:out)'" ),
        "This error *might* be caused by importing numpy instead of autograd.numpy. \n"
        "Check that you have 'import autograd.numpy as np' instead of 'import numpy as np'."),
    ((AttributeError, r"object has no attribute" ),
        "This error *might* be caused by importing numpy instead of autograd.numpy,"
        "or otherwise using a raw numpy function instead of the autograd-wrapped version. \n"
        "Check that you have 'import autograd.numpy as np' instead of 'import numpy as np'."),
]

defgrad_deprecated = \
'''
------------------------------
    defgrad is deprecated!
------------------------------
Use defvjp instead ("define vector-Jacobian product").
The interface is a little different - look at
autograd/numpy/numpy_grads.py for examples.
'''

def add_error_hints(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    try: return fun(*args, **kwargs)
    except Exception as e: add_extra_error_message(e)
  return wrapped

def check_common_errors(error_type, error_message):
    keys, vals = zip(*common_errors)
    matches = [error_type == key[0]
               and len(re.findall(key[1], error_message)) != 0
               for key in keys]
    num_matches = sum(matches)

    if num_matches == 1:
        return vals[matches.index(True)]

def add_extra_error_message(e):
    etype, value, traceback = sys.exc_info()
    extra_message = check_common_errors(type(e), str(e))

    if extra_message:
        if sys.version_info >= (3,):
            raise_from(AutogradHint(extra_message), e)
        else:
            raise_(AutogradHint, (extra_message, etype, value), traceback)
    raise_(etype, value, traceback)
