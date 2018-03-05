import autograd.fmap_util as fu

def test_limited_fmap():
    xs = [(1,2), {'x' : 3}, 4]
    cond = [(True, False), {'x' : False}, True]
    lfmap = fu.limited_fmap(fu.container_fmap, cond)
    assert lfmap(lambda x: x * 10, xs) == [(10, 2), {'x' : 3}, 40]

def test_fmap_to_list():
    assert fu.fmap_to_list(fu.container_fmap,
                           [(1,2), {'x' : 3}, 4]) == [1, 2, 3, 4]

def test_fmap_to_basis():
    basis_hidden_type = fu.fmap_to_basis(fu.container_fmap,
                                    [(1,2), {'x' : 3}, 4])
    basis = fu.container_fmap(lambda x: x.value, basis_hidden_type)
    assert basis == [
        (        [(True , False), {'x' : False}, False],
                 [(False, True ), {'x' : False}, False] ),
        { 'x' :  [(False, False), {'x' : True }, False] },
                 [(False, False), {'x' : False}, True ]
        ]


def test_container_fmap():
    xs = [(1,2), {'x' : 3}, 4]
    ys = [(10,20), {'x' : 30}, 40]
    zs = [(11,22), {'x' : 33}, 44]
    assert fu.container_fmap(lambda x, y: x + y, xs, ys) == zs
