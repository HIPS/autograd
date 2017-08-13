class TailCall(object):
    def __init__(self, f, *args, **kwargs):
        if isinstance(f, trampoline):
            kwargs['trampoline_parent'] = True
        self.thunk = lambda: f(*args, **kwargs)

class trampoline(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, *args, **kwargs):
        trampoline_parent = kwargs.pop('trampoline_parent', False)
        if trampoline_parent:
            return self.f(*args, **kwargs)

        ans = TailCall(self.f, *args, **kwargs)
        while isinstance(ans, TailCall):
            ans = ans.thunk()
        return ans
