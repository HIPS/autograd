from __future__ import absolute_import
import cupy.linalg as cpla
from .cupy_wrapper import wrap_namespace
from . import cupy_wrapper as acp

wrap_namespace(cpla.__dict__, globals())


def grad_norm(g, ans, vs, gvs, x, ord=None, axis=None):
    def check_implemented():
        matrix_norm = (x.ndim == 2 and axis is None) or isinstance(axis, tuple)

        if matrix_norm:
            if not (ord is None or ord == 'fro' or ord == 'nuc'):
                raise NotImplementedError('Gradient of matrix norm not '
                                          'implemented for ord={}'.format(ord))
        elif not (ord is None or ord > 1):
            raise NotImplementedError('Gradient of norm not '
                                      'implemented for ord={}'.format(ord))

    if axis is None:
        expand = lambda a: a
    elif isinstance(axis, tuple):
        row_axis, col_axis = axis
        if row_axis > col_axis:
            row_axis = row_axis - 1
        expand = lambda a: acp.expand_dims(acp.expand_dims(a,
                                                   row_axis), col_axis)
    else:
        expand = lambda a: acp.expand_dims(a, axis=axis)

    if ord == 'nuc':
        if axis is None:
            roll = lambda a: a
            unroll = lambda a: a
        else:
            row_axis, col_axis = axis
            if row_axis > col_axis:
                row_axis = row_axis - 1
            # Roll matrix axes to the back
            roll = lambda a: acp.rollaxis(acp.rollaxis(a, col_axis, a.ndim),
                                          row_axis, a.ndim-1)
            # Roll matrix axes to their original position
            unroll = lambda a: acp.rollaxis(acp.rollaxis(a, a.ndim-2, row_axis),
                                            a.ndim-1, col_axis)

    check_implemented()
    if ord is None or ord == 2 or ord is 'fro':
        return expand(g / ans) * x
    elif ord == 'nuc':
        dot = acp.dot if x.ndim == 2 else partial(acp.einsum, '...ij,...jk->...ik')
        x_rolled = roll(x)
        u, s, vt = svd(x_rolled, full_matrices=False)
        uvt_rolled = dot(u, vt)
        # Roll the matrix axes back to their correct positions
        uvt = unroll(uvt_rolled)
        g = expand(g)
        return g * uvt
    else:
        # see https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
        return expand(g / ans**(ord-1)) * x * acp.abs(x)**(ord-2)
norm.defvjp(grad_norm)
