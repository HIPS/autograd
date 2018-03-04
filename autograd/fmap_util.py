

def limited_fmap(fmap, conds):
    """Only touches elements for which cond is true."""
    def lfmap(f, *args):
        new_f = lambda cond, x, *rest: f(x, *rest) if cond else x
        return fmap(new_f, conds, *args)

    return lfmap

def fmap_to_list(fmap, xs):
    L = []
    fmap(L.append, xs)
    return L

def fmap_to_zipped(fmap, xs, ys):
    L = []
    fmap(lambda x, y: L.append((x, y)), xs, ys)
    return L
