import sys
from .errors import add_extra_error_message
from future.utils import raise_

def unary_to_nary(unary_operator):
    @wraps(unary_operator)
    def nary_operator(fun, argnum=0, **nary_op_kwargs):
        @attach_name_and_doc(fun, argnum, unary_operator)
        def nary_f(*args, **kwargs):
            def unary_f(x):
                try:
                    if isinstance(argnum, int):
                        subargs = subvals(args, [(argnum, x)])
                    else:
                        subargs = subvals(args, zip(argnum, x))
                    return fun(*subargs, **kwargs)
                except Exception as e:
                    raise_(*add_extra_error_message(e))
            if isinstance(argnum, int):
                x = args[argnum]
            else:
                x = tuple(args[i] for i in argnum)
            return unary_operator(unary_f, x, **nary_op_kwargs)
        return nary_f
    return nary_operator

def attach_name_and_doc(fun, argnum, op):
    fname = lambda f: getattr(f, '__name__', '[unknown name]')
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """
    def wrap(f):
        data = dict(op=fname(op), fun=fname(fun), argnum=argnum)
        f.__name__ = namestr.format(**data)
        f.__doc__  = docstr.format(**data)
        return f
    return wrap

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def subval(x, i, v):
    x_ = list(x)
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
