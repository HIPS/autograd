from __future__ import absolute_import
import numpy as onp
import operator as op

from autograd.core import primitive, getval, vspace
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, take, array_types
from builtins import range, zip
from future.utils import string_types

# ----- Functions that are constant w.r.t. continuous inputs -----

anp.where.defgrad_is_zero(argnums=(0,))
anp.nan_to_num.defgrad(lambda g, ans, vs, gvs, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs -----

anp.add.defgrad(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.add.defgrad(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g), argnum=1)
anp.multiply.defgrad(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, y * g))
anp.multiply.defgrad(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, x * g), argnum=1)
anp.subtract.defgrad(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.subtract.defgrad(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g), argnum=1)
anp.divide.defgrad(     lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs,   g / y))
anp.divide.defgrad(     lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
anp.power.defgrad(      lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * y * x ** (y - 1)))
anp.power.defgrad(      lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * anp.log(x) * x ** y), argnum=1)
anp.maximum.defgrad(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * (x == ans)))
anp.maximum.defgrad(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * (y == ans)), argnum=1)
anp.minimum.defgrad(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * (x == ans)))
anp.minimum.defgrad(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * (y == ans)), argnum=1)
anp.logaddexp.defgrad(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * anp.exp(x-ans)))
anp.logaddexp.defgrad(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * anp.exp(y-ans)), argnum=1)
anp.logaddexp2.defgrad( lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * 2**(x-ans)))
anp.logaddexp2.defgrad( lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * 2**(y-ans)), argnum=1)
anp.true_divide.defgrad(lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g / y))
anp.true_divide.defgrad(lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
anp.mod.defgrad(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.remainder.defgrad(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.mod.defgrad(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)
anp.remainder.defgrad(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)

# ----- Simple grads -----

anp.negative.defgrad(lambda g, ans, vs, gvs, x: -g)
anp.abs.defgrad(     lambda g, ans, vs, gvs, x : g * anp.conj(x) / ans)
anp.fabs.defgrad(    lambda g, ans, vs, gvs, x : anp.sign(x) * g)  # fabs doesn't take complex numbers.
anp.absolute.defgrad(lambda g, ans, vs, gvs, x : g * anp.conj(x) / ans)
anp.reciprocal.defgrad(lambda g, ans, vs, gvs, x : - g / x**2)
anp.exp.defgrad(   lambda g, ans, vs, gvs, x : ans * g)
anp.exp2.defgrad(  lambda g, ans, vs, gvs, x : ans * anp.log(2) * g)
anp.expm1.defgrad( lambda g, ans, vs, gvs, x : (ans + 1) * g)
anp.log.defgrad(   lambda g, ans, vs, gvs, x : g / x)
anp.log2.defgrad(  lambda g, ans, vs, gvs, x : g / x / anp.log(2))
anp.log10.defgrad( lambda g, ans, vs, gvs, x : g / x / anp.log(10))
anp.log1p.defgrad( lambda g, ans, vs, gvs, x : g / (x + 1))
anp.sin.defgrad(   lambda g, ans, vs, gvs, x : g * anp.cos(x))
anp.cos.defgrad(   lambda g, ans, vs, gvs, x : - g * anp.sin(x))
anp.tan.defgrad(   lambda g, ans, vs, gvs, x : g / anp.cos(x) **2)
anp.arcsin.defgrad(lambda g, ans, vs, gvs, x : g / anp.sqrt(1 - x**2))
anp.arccos.defgrad(lambda g, ans, vs, gvs, x :-g / anp.sqrt(1 - x**2))
anp.arctan.defgrad(lambda g, ans, vs, gvs, x : g / (1 + x**2))
anp.sinh.defgrad(  lambda g, ans, vs, gvs, x : g * anp.cosh(x))
anp.cosh.defgrad(  lambda g, ans, vs, gvs, x : g * anp.sinh(x))
anp.tanh.defgrad(  lambda g, ans, vs, gvs, x : g / anp.cosh(x) **2)
anp.arcsinh.defgrad(lambda g, ans, vs, gvs, x : g / anp.sqrt(x**2 + 1))
anp.arccosh.defgrad(lambda g, ans, vs, gvs, x : g / anp.sqrt(x**2 - 1))
anp.arctanh.defgrad(lambda g, ans, vs, gvs, x : g / (1 - x**2))
anp.rad2deg.defgrad(lambda g, ans, vs, gvs, x : g / anp.pi * 180.0)
anp.degrees.defgrad(lambda g, ans, vs, gvs, x : g / anp.pi * 180.0)
anp.deg2rad.defgrad(lambda g, ans, vs, gvs, x : g * anp.pi / 180.0)
anp.radians.defgrad(lambda g, ans, vs, gvs, x : g * anp.pi / 180.0)
anp.square.defgrad( lambda g, ans, vs, gvs, x : g * 2 * x)
anp.sqrt.defgrad(   lambda g, ans, vs, gvs, x : g * 0.5 * x**-0.5)
anp.sinc.defgrad(   lambda g, ans, vs, gvs, x : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
anp.reshape.defgrad(lambda g, ans, vs, gvs, x, shape, order=None : anp.reshape(g, vs.shape, order=order))
anp.roll.defgrad(   lambda g, ans, vs, gvs, x, shift, axis=None  : anp.roll(g, -shift, axis=axis))
anp.array_split.defgrad(lambda g, ans, vs, gvs, ary, idxs, axis=0 : anp.concatenate(g, axis=axis))
anp.split.defgrad(      lambda g, ans, vs, gvs, ary, idxs, axis=0 : anp.concatenate(g, axis=axis))
anp.vsplit.defgrad(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=0))
anp.hsplit.defgrad(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=1))
anp.dsplit.defgrad(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=2))
anp.ravel.defgrad(  lambda g, ans, vs, gvs, x, order=None   : anp.reshape(g, vs.shape, order=order))
anp.expand_dims.defgrad(lambda g, ans, vs, gvs, x, axis     : anp.reshape(g, vs.shape))
anp.squeeze.defgrad(lambda g, ans, vs, gvs, x, axis=None    : anp.reshape(g, vs.shape))
anp.diag.defgrad(   lambda g, ans, vs, gvs, x, k=0          : anp.diag(g, k))
anp.flipud.defgrad( lambda g, ans, vs, gvs, x,              : anp.flipud(g))
anp.fliplr.defgrad( lambda g, ans, vs, gvs, x,              : anp.fliplr(g))
anp.rot90.defgrad(  lambda g, ans, vs, gvs, x, k=1          : anp.rot90(g, -k))
anp.trace.defgrad(  lambda g, ans, vs, gvs, x, offset=0     :
                    anp.einsum('ij,...->ij...', anp.eye(x.shape[0], x.shape[1], k=offset), g))
anp.full.defgrad(   lambda g, ans, vs, gvs, shape, fill_value, dtype=None : anp.sum(g), argnum=1)
anp.triu.defgrad(   lambda g, ans, vs, gvs, x, k=0          : anp.triu(g, k=k))
anp.tril.defgrad(   lambda g, ans, vs, gvs, x, k=0          : anp.tril(g, k=k))
anp.clip.defgrad(   lambda g, ans, vs, gvs, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
anp.swapaxes.defgrad(lambda g, ans, vs, gvs, x, axis1, axis2: anp.swapaxes(g, axis2, axis1))
anp.rollaxis.defgrad(lambda g, ans, vs, gvs, a, axis, start=0: anp.rollaxis(g, start - 1, axis) if start > axis
                                                 else anp.rollaxis(g, start, axis + 1))
anp.real_if_close.defgrad(lambda g, ans, vs, gvs, x : match_complex(vs, g))
anp.real.defgrad(  lambda g, ans, vs, gvs, x   : match_complex(vs, g))
anp.imag.defgrad(  lambda g, ans, vs, gvs, x   : match_complex(vs, -1j * g))
anp.conj.defgrad(  lambda g, ans, vs, gvs, x   : anp.conj(g))
anp.angle.defgrad( lambda g, ans, vs, gvs, x   : match_complex(vs, g * anp.conj(x * 1j) / anp.abs(x)**2))
anp.where.defgrad( lambda g, ans, vs, gvs, c, x=None, y=None : anp.where(c, g, anp.zeros(g.shape)), argnum=1)
anp.where.defgrad( lambda g, ans, vs, gvs, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g), argnum=2)
anp.cross.defgrad(lambda g, ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(b, g, axisb, axisc, axisa, axis), argnum=0)
anp.cross.defgrad(lambda g, ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(g, a, axisc, axisa, axisb, axis), argnum=1)

# ----- Trickier grads -----

def grad_diff(g, ans, vs, gvs, a, n=1, axis=-1):
    nd = len(vs.shape)
    sl1 = [slice(None)]*nd
    sl1[axis] = slice(None, 1)

    sl2 = [slice(None)]*nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return anp.concatenate((-g[sl1], -anp.diff(g, axis=axis), g[sl2]), axis=axis)
        shape = list(gvs.shape)
        shape[axis] = 1
        return anp.zeros(shape)

    def helper(g, n):
        if n == 0:
            return g
        return helper(undiff(g), n-1)
    return helper(g, n)

anp.diff.defgrad(grad_diff)

def grad_repeat(g, ans, vs, gvs, x, repeats, axis=None):
    shape = x.shape
    if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
        expanded = anp.reshape(g, (anp.prod(shape),) + (repeats,))
        return anp.reshape(anp.sum(expanded, axis=1, keepdims=False), shape)
    else:
        if shape[axis] == 1:  # For this common case, the logic is simple.
            return anp.sum(g, axis=axis, keepdims=True)
        else:
            expanded = anp.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
            return anp.sum(expanded, axis=axis+1, keepdims=False)
anp.repeat.defgrad(grad_repeat)

def grad_tile(g, ans, vs, gvs, x, reps):
    reps = [reps] if anp.isscalar(reps) else reps
    for axis, rep in enumerate(reps):
        g = sum(anp.split(g, rep, axis))
    return anp.reshape(g, x.shape)
anp.tile.defgrad(grad_tile)

def grad_transpose(g, ans, vs, gvs, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return anp.transpose(g, axes)
anp.transpose.defgrad(grad_transpose)

def repeat_to_match_shape(g, vs, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    shape = vs.shape
    if shape == ():
        return g, 1
    elif axis is None:
        if keepdims:
            g = anp.sum(g)
        return anp.full(shape, g, dtype=vs.dtype), anp.prod(shape)
    elif isinstance(axis, int):
        if not keepdims:
            g = anp.expand_dims(g, axis)
        return anp.repeat(g, shape[axis], axis), shape[axis]
    elif isinstance(axis, tuple):
        repeats  = [shape[i] if i in axis else 1 for i in range(len(shape))]
        expanded = [shape[i] if i not in axis else 1 for i in range(len(shape))]
        num_reps = anp.prod(anp.array(shape)[list(axis)])
        if not keepdims:
            g = anp.reshape(g, expanded)
        return anp.tile(g, repeats), num_reps
    else:
        raise Exception('Axis type {} not valid'.format(type(axis)))

def grad_np_sum(g, ans, vs, gvs, x, axis=None, keepdims=False):
    return repeat_to_match_shape(g, vs, axis, keepdims)[0]
anp.sum.defgrad(grad_np_sum)

def grad_np_mean(g, ans, vs, gvs, x, axis=None, keepdims=False):
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
    return g_repeated / num_reps
anp.mean.defgrad(grad_np_mean)

def grad_np_prod(g, ans, vs, gvs, x, axis=None, keepdims=False): # TODO: Support tuples of axes.
    g_repeated, _ = repeat_to_match_shape(g * ans, vs, axis, keepdims)
    return g_repeated / x
anp.prod.defgrad(grad_np_prod)

def grad_np_var(g, ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    if vs.iscomplex:
        g = g + 0j
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)
anp.var.defgrad(grad_np_var)

def make_grad_np_std(g, ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    if vs.iscomplex:
        g = g + 0j
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)  # Avoid division by zero.
    if num_reps <= 1:
        return g_repeated * 0.0
    else:
        g_repeated, num_reps = repeat_to_match_shape(g / ans, vs, axis, keepdims)
        x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
        return g_repeated * x_minus_mean / (num_reps - ddof)
anp.std.defgrad(make_grad_np_std)

def grad_chooser(g, ans, vs, gvs, x, axis=None, keepdims=None):
    """Builds gradient of functions that choose a single item, such as min or max."""
    g_repeated, _ = repeat_to_match_shape(g, vs, axis, keepdims)
    argmax_locations = x == repeat_to_match_shape(ans, vs, axis, keepdims)[0]
    return g_repeated * argmax_locations
anp.max.defgrad(grad_chooser)
anp.min.defgrad(grad_chooser)
anp.amax.defgrad(grad_chooser)
anp.amin.defgrad(grad_chooser)

def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1,...]
    return x.swapaxes(0, axis)

def grad_np_cumsum(g, ans, vs, gvs, x, axis=None):
    if axis:
        return reverse_axis(anp.cumsum(reverse_axis(g, axis)), axis)
    else:
        return anp.reshape(anp.cumsum(g[::-1], axis)[::-1], x.shape)
anp.cumsum.defgrad(grad_np_cumsum)

def grad_dot(argnum, g, ans, vs, gvs, A, B):
    A_ndim, B_ndim = anp.ndim(A), anp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [max(0, B_ndim - 2)])
    return grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=axes)
anp.dot.defgrads(grad_dot, [0, 1])

def grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=2):
    A_ndim = anp.ndim(A)
    B_ndim = anp.ndim(B)
    g_ndim = len(gvs.shape)
    if type(axes) is int:
        if axes > 0:
            axes = (list(range(A_ndim))[-axes:],
                    list(range(B_ndim))[:axes])
        else:
            axes = [(), ()] # summing over zero axes

        assert len(axes[0]) == len(axes[1])  # required by tensordot

    N_axes_summed = len(axes[0])
    if argnum == 0:
        X, Y = A, B
        X_ndim, Y_ndim = A_ndim, B_ndim
        X_axes_summed, Y_axes_summed = axes
        g_axes_from_Y = list(range(g_ndim))[(X_ndim - N_axes_summed):]
    else:
        X, Y = B, A
        X_ndim, Y_ndim = B_ndim, A_ndim
        X_axes_summed, Y_axes_summed = axes[::-1]
        g_axes_from_Y = list(range(g_ndim))[:(Y_ndim - N_axes_summed)]

    Y_axes_ignored = [i for i in range(Y_ndim) if i not in Y_axes_summed]
    result = anp.tensordot(g, Y, axes=[g_axes_from_Y, Y_axes_ignored])
    sorted_axes_pairs = sorted(zip(X_axes_summed, Y_axes_summed), key =lambda x : x[1])
    forward_permutation = ([i for i in range(X_ndim) if i not in X_axes_summed]
                         + [i for i, _ in sorted_axes_pairs])
    reverse_permutation = list(anp.argsort(forward_permutation))
    if result.ndim == 0:
        result = result[()]
    return anp.transpose(result, axes=reverse_permutation)
anp.tensordot.defgrads(grad_tensordot, [0, 1])

anp.outer.defgrad(lambda g, ans, vs, gvs, a, b : anp.dot(g, b.T))
anp.outer.defgrad(lambda g, ans, vs, gvs, a, b : anp.dot(a.T, g), argnum=1)

def grad_concatenate_args(argnum, g, ans, vs, gvs, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    start = sum([a.shape[axis] for a in args[:argnum-1]])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + args[argnum-1].shape[axis])
    return take(g, idxs)
anp.concatenate_args.grad = grad_concatenate_args

def wrapped_reshape(x, *args, **kwargs):
    # The reshape method can be called like A.reshape((5,4)) or A.reshape(5,4).
    # The reshape function doesn't support both ways, so we have to wrap it.
    if isinstance(args[0], int):
        return anp.reshape(x, args, **kwargs)
    else:
        return anp.reshape(x, *args, **kwargs)
setattr(ArrayNode, 'reshape', wrapped_reshape)

def grad_sort(g, ans, vs, gvs, x, axis=-1, kind='quicksort', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays.")
    sort_perm = anp.argsort(x, axis, kind, order)
    return unpermuter(g, sort_perm)
anp.sort.defgrad(grad_sort)
anp.msort.defgrad(grad_sort)  # Until multi-D is allowed, these are the same.

def grad_partition(g, ans, vs, gvs, x, kth, axis=-1, kind='introselect', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays.")
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return unpermuter(g, partition_perm)
anp.partition.defgrad(grad_partition)

def unpermuter(g, permutation):
    unsort = anp.zeros(len(permutation), dtype=int)
    unsort[permutation] = list(range(len(permutation)))
    return g[unsort]

def grad_reshape_list(g, ans, vs, gvs, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")
    return anp.reshape(g, anp.shape(arys[0]))
anp.atleast_1d.defgrad(grad_reshape_list)
anp.atleast_2d.defgrad(grad_reshape_list)
anp.atleast_3d.defgrad(grad_reshape_list)

def grad_einsum(argnum, g, ans, vs, gvs, operands, kwargs):
    # Gradient of einsum is obtained by swapping outgrad with the argument
    # being differentiated wrt.
    if isinstance(operands[0], string_types):  # using "ijk" convention.
        subscripts, operands = operands[0], operands[1:]
        if not '->' in subscripts:
            raise NotImplementedError("Need indices on both sides.")
        op_num = argnum - 1
        input_subs, output_subs = subscripts.split('->')
        input_subs_list = input_subs.split(',')
        subs_wrt = input_subs_list[op_num]
        rest_of_ops = operands[:op_num] + operands[op_num + 1:]
        rest_of_subs = input_subs_list[:op_num] + input_subs_list[op_num + 1:]
        new_subscripts = ','.join([output_subs] + rest_of_subs) + '->' + subs_wrt
        return anp.einsum(new_subscripts, *((g,) + rest_of_ops))
    else:  # Using (op0, sublist0, op1, sublist1..., sublistout) convention.
        if len(operands) % 2 == 0:
            raise NotImplementedError("Need sublistout argument")
        operands = list(operands)
        rest_of_ops = [operands[-1]] + operands[:argnum] + operands[(argnum+2):-1] + [operands[argnum+1]]
        return anp.einsum(g, *rest_of_ops)

anp.einsum.grad = grad_einsum

@primitive
def make_diagonal(D, offset=0, axis1=0, axis2=1):
    # Numpy doesn't offer a complement to np.diagonal: a function to create new
    # diagonal arrays with extra dimensions. We need such a function for the
    # gradient of np.diagonal and it's also quite handy to have. So here it is.
    if not (offset==0 and axis1==-1 and axis2==-2):
        raise NotImplementedError("Currently make_diagonal only supports offset=0, axis1=-1, axis2=-2")

    # We use a trick: calling np.diagonal returns a view on the original array,
    # so we can modify it in-place. (only valid for numpy version >= 1.10.)
    new_array = onp.zeros(D.shape + (D.shape[-1],))
    new_array_diag = onp.diagonal(new_array, offset=0, axis1=-1, axis2=-2)
    new_array_diag.flags.writeable = True
    new_array_diag[:] = D
    return new_array

anp.make_diagonal = make_diagonal
anp.diagonal.defgrad(
    lambda g, ans, vs, gvs, A, offset=0, axis1=0, axis2=1 :
    anp.make_diagonal(g, offset, axis1, axis2))
anp.make_diagonal.defgrad(
    lambda g, ans, vs, gvs, D, offset=0, axis1=0, axis2=1 :
    anp.diagonal(g, offset, axis1, axis2))

def match_complex(vs, x):
    x_iscomplex = vspace(getval(x)).iscomplex
    if x_iscomplex and not vs.iscomplex:
        return anp.real(x)
    elif not x_iscomplex and vs.iscomplex:
        return x + 0j
    else:
        return x

def unbroadcast(vs, gvs, result):
    while anp.ndim(result) > len(vs.shape):
        result = anp.sum(result, axis=0)
    for axis, size in enumerate(vs.shape):
        if size == 1:
            result = anp.sum(result, axis=axis, keepdims=True)
    if gvs.iscomplex and not vs.iscomplex:
        result = anp.real(result)
    return result
