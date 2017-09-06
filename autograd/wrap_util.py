from .util import subvals

def unary_to_nary(unary_operator):
    @wraps(unary_operator)
    def nary_operator(fun, argnum=0, *nary_op_args, **nary_op_kwargs):
        assert type(argnum) in (int, tuple, list), argnum
        @wrap_nary_f(fun, unary_operator, argnum)
        def nary_f(*args, **kwargs):
            @wraps(fun)
            def unary_f(x):
                if isinstance(argnum, int):
                    subargs = subvals(args, [(argnum, x)])
                else:
                    subargs = subvals(args, zip(argnum, x))
                return fun(*subargs, **kwargs)
            if isinstance(argnum, int):
                x = args[argnum]
            else:
                x = tuple(args[i] for i in argnum)
            return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)
        return nary_f
    return nary_operator

def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):
    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun), **kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun), doc=get_doc(fun), **kwargs)
        finally:
            return f
    return _wraps

def wrap_nary_f(fun, op, argnum):
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """
    return wraps(fun, namestr, docstr, op=get_name(op), argnum=argnum)

get_name = lambda f: getattr(f, '__name__', '[unknown name]')
get_doc  = lambda f: getattr(f, '__doc__' , '')
