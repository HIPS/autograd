from itertools import count

def apply(f, *args):
    return f(*args)  # built-in version doesn't do *args (and missing in py3)

def maybe_map(f, xs, *rest):
    if isinstance(xs, (tuple, list)):
        return map(f, xs, *rest)
    else:
        return f(xs, *rest)

def select_map(idxs):
    def smap(f, *args):
        out = list(args[0])
        new_vals = map(f, *[[xs[i] for i in idxs] for xs in args])
        for i, v in zip(idxs, new_vals):
            out[i] = v
        return out
    return smap

def compose_fmaps(fmap1, fmap2):
    return lambda f, *a: fmap1(lambda *a: fmap2(f, *a),  *a)

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

def fmap_to_basis(fmap, xs):
    counter = count()
    idxs = fmap(lambda _: counter.next(), xs)
    return fmap(lambda i: HideType(fmap(lambda j: i==j, idxs)), idxs)

def container_fmap(f, xs, *rest):
    # TODO(dougalm): consider making this extensible
    t = type(xs)
    if t in (tuple, list):
        return t(map(lambda x, *rest_elts:
                     container_fmap(f, x, *rest_elts), xs, *rest))
    elif t is dict:
        return {k : container_fmap(f, v, *[r[k] for r in rest])
                for k, v in xs.items()}
    else:
        return f(xs, *rest)

class HideType(object):
    def __init__(self, value):
        self.value = value
