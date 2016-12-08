import autograd.numpy as np
import autograd.numpy.random as npr
import itertools as it
from autograd.core import vspace, vspace_flatten
from functools import partial
import autograd.core as agc
import operator as op

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

def concat(lists):
    return list(it.chain(*lists))

def numerical_jacobian(fun, argnum, args, kwargs):
    def vector_fun(x):
        args_tmp = list(args)
        args_tmp[argnum] = vs_in.unflatten(vs_in.flatten(args[argnum]) + x)
        return vs_out.flatten(fun(*args_tmp, **kwargs))

    vs_in  = vspace(args[argnum])
    vs_out = vspace(fun(*args, **kwargs))
    return np.stack([(vector_fun(dx) - vector_fun(-dx)) / EPS
                     for dx in np.eye(vs_in.size) * EPS / 2]).T

def check_args(fun, argnum, args, kwargs):
    ans = fun(*args)
    in_vspace  = vspace(args[argnum])
    ans_vspace = vspace(ans)
    jac = numerical_jacobian(fun, argnum, args, kwargs)
    for outgrad in ans_vspace.examples():
        result = fun.vjps[argnum](
            outgrad, ans, in_vspace, ans_vspace, *args, **kwargs)
        result_vspace = vspace(result)
        result_reals = vspace_flatten(result, True)
        nd_result_reals = np.dot(vspace_flatten(outgrad, True), jac)
        assert result_vspace == in_vspace, \
            report_mismatch(fun, argnum, args, kwargs, outgrad,
                            in_vspace, result_vspace)
        assert np.allclose(result_reals, nd_result_reals),\
            report_nd_failure(fun, argnum, args, kwargs, outgrad,
                              nd_result_reals, result_reals)

def check_primitive(fun, vspace_instances, kwargs={}, argnums=[0]):
    arg_sets = [concat([vs.examples() for vs in vsi])
                for vsi in vspace_instances]
    for argnum, args in it.product(argnums, it.product(*arg_sets)):
        check_args(fun, argnum, args, kwargs)

array_shapes = [(), (1,), (1,1), (2,), (3,1), (1,2), (3,2), (2,3,1)]
real_arrays    = [vspace(np.zeros(s)               ) for s in array_shapes]
complex_arrays = [vspace(np.zeros(s, dtype=complex)) for s in array_shapes]
composite_values = [[0.0], [0.0, np.zeros((2,1))],
                    [0.0, np.zeros((2,1)), [0.0]]]
lists  = list(map(vspace, composite_values))
tuples = list(map(vspace, map(tuple, composite_values)))
dicts  = list(map(vspace, [dict(zip(it.count(), x)) for x in composite_values]))

all_arrays  = real_arrays + complex_arrays
everything  = all_arrays + lists + tuples + dicts

report_mismatch = \
'''
Vspace mismatch
(function {} argnum {} args {}, kwargs {}, outgrad {}):
expected {}
got      {}
'''.format

report_nd_failure = \
'''
Numerical derivative mismatch
(function {} argnum {} args {}, kwargs {}, , outgrad {}):
expected {}
got      {}

'''.format

report_flatten_unflatten = \
'''{} flatten/unflatten failed.
expected {}
got      {}'''.format

def test_flatten_unflatten():
    for vs in everything:
        v = npr.randn(vs.size)
        v2 = vs.flatten(vs.unflatten(v))
        assert np.all(v2 == v), \
            report_flatten_unflatten(vs, v, v2)

def test_identity(): check_primitive(agc.identity, (everything,))
def test_sin():      check_primitive(np.sin,       (all_arrays,))
def test_np_sum():   check_primitive(np.add, (all_arrays, all_arrays), argnums=[0,1])
