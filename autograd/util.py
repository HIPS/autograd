import sys
from .errors import add_error_hints

def unary_to_nary(unary_function_modifier):
    def nary_function_modifier(nary_f_in, argnum=0):
        @attach_name_and_doc(nary_f_in, argnum, unary_function_modifier)
        @add_error_hints
        def nary_f_out(*args, **kwargs):
            def unary_f_in(x):
                _args = list(args)
                _args[argnum] = x
                return nary_f_in(*_args, **kwargs)
            return unary_function_modifier(unary_f_in)(args[argnum])
        return nary_f_out
    return nary_function_modifier

def attach_name_and_doc(fun, argnum, op):
    fname = lambda f: getattr(fun, '__name__', '[unknown name]')
    namestr = "{0}_{1}_wrt_argnum_{2}".format(fname(op), fname(fun), argnum)
    docstr = ("{0} of function {1} with respect to argument number {2}. "
              "Has the same arguments as {1} but the return value has type of "
              "argument {2}.".format(fname(op), fname(fun), argnum))

    def wrap(gradfun):
        try:
            gradfun.__name__ = namestr
            gradfun.__doc__ = docstr
        finally:
            return gradfun
    return wrap

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def wraps(f_raw):
    def wrap(f_wrapped):
        try:
            f_wrapped.__name__ = f_raw.__name__
            f_wrapped.__doc__  = f_raw.__doc__
        finally:
            return f_wrapped
    return wrap

if sys.version_info >= (3,):
    def func(f): return f
else:
    def func(f): return f.im_func
