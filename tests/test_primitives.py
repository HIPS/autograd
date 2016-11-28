import numpy as np
import numpy.random as npr
import itertools as it
from autograd.core import vspace, flatten
from functools import partial
import autograd.core as agc

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

def concat(lists):
    return list(it.chain(*lists))

def numerical_jacobian(fun, argnum, args, kwargs):
    def vector_fun(x):
        args_tmp = list(args)
        args_tmp[argnum] = vs.unflatten(vs.flatten(args[argnum]) + x)
        result = fun(*args_tmp, **kwargs)
        return vs.flatten(result)

    vs = vspace(args[argnum])
    N_in  = vs.size
    N_out = vspace(fun(*args, **kwargs)).size
    jac = np.zeros((N_out, N_in))

    for i in range(N_in):
        dx = np.zeros(N_in)
        dx[i] = EPS / 2
        jac[:, i] = (vector_fun(dx) - vector_fun(-dx)) / EPS

    return jac

def check_args(fun, argnum, args, kwargs):
    ans = fun(*args)
    in_vspace  = vspace(args[argnum])
    ans_vspace = vspace(ans)
    jac = numerical_jacobian(fun, argnum, args, kwargs)
    for outgrad in ans_vspace.examples():
        result = fun.grads[argnum](outgrad, ans, *args, **kwargs)
        result_vspace = vspace(result)
        result_reals = flatten(result)
        nd_result_reals = np.dot(flatten(outgrad), jac)
        assert result_vspace == in_vspace, \
            report_mismatch(fun, argnum, args, kwargs, outgrad,
                            in_vspace, result_vspace)
        assert np.allclose(result_reals, nd_result_reals),\
            report_nd_failure(fun, argnum, args, kwargs, outgrad,
                              result_reals, nd_result_reals)

def check_primitive(fun, argnums, vspace_instances, kwargs):
    arg_sets = [concat([vs.examples() for vs in vsi])
                for vsi in vspace_instances]
    for argnum, args in it.product(argnums, it.product(*arg_sets)):
        check_args(fun, argnum, args, kwargs)

scalars = [vspace(0.0)]
complex_scalars = [vspace(0.0 + 0.0j)]
array_shapes = [(), (1,), (1,1), (2,), (2,1), (1,2), (2,3), (2,3,4)]
arrays  = [vspace(np.zeros(s)) for s in array_shapes]
complex_arrays = [vspace(np.zeros(s, dtype=complex)) for s in array_shapes]
composite_values = [[], [0.0], [0.0, np.zeros((2,1))],
                    [0.0, np.zeros((2,1)), [0.0]]]
lists  = map(vspace, composite_values)
tuples = map(vspace, map(tuple, composite_values))
dicts  = map(vspace, [dict(zip(it.count(), x)) for x in composite_values])

all_scalars = scalars + complex_scalars
all_arrays  = arrays + complex_arrays
everything  = all_scalars + all_arrays + lists + dicts

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

def test_identity(): check_primitive(agc.identity, [0], (everything,), {})
