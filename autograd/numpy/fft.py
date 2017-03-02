from __future__ import absolute_import
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_grads import match_complex
from . import numpy_wrapper as anp
from autograd.core import primitive
from builtins import zip

wrap_namespace(ffto.__dict__, globals())

# TODO: make fft gradient work for a repeated axis,
# e.g. by replacing fftn with repeated calls to 1d fft along each axis
def fft_defvjp(fft_fun):
    def fft_grad(g, ans, vs, gvs, x, *args, **kwargs):
        check_no_repeated_axes(*args, **kwargs)
        return match_complex(vs, truncate_pad(fft_fun(g, *args, **kwargs), vs.shape))
    fft_fun.defvjp(fft_grad)

for fft_fun in (fft, ifft, fft2, ifft2, fftn, ifftn):
    fft_defvjp(fft_fun)

fftshift.defvjp( lambda g, ans, vs, gvs, x, axes=None : match_complex(vs, anp.conj(ifftshift(anp.conj(g), axes))))
ifftshift.defvjp(lambda g, ans, vs, gvs, x, axes=None : match_complex(vs, anp.conj(fftshift(anp.conj(g), axes))))

def rfft_defvjp(rfft_fun, irfft_fun, n):
    def rfft_grad(g, ans, vs, gvs, x, *args, **kwargs):
        nd = x.ndim
        check_no_repeated_axes(*args, **kwargs)
        axes = find_axes(nd, n, *args, **kwargs)
        check_even_shape(axes, vs.shape)
        fac, norm = make_rfft_factors(axes, gvs.shape, vs.shape)
        g = anp.conj(g / fac)
        r = match_complex(vs, truncate_pad((irfft_fun(g, *args, **kwargs)), vs.shape))
        return r * norm

    rfft_fun.defvjp(rfft_grad)

    def irfft_grad(g, ans, vs, gvs, x, *args, **kwargs):
        check_no_repeated_axes(*args, **kwargs)
        nd = x.ndim
        axes = find_axes(nd, n, *args, **kwargs)
        check_even_shape(axes, gvs.shape)
        r = match_complex(vs, truncate_pad((rfft_fun(g, *args, **kwargs)), vs.shape))
        fac, norm = make_rfft_factors(axes, vs.shape, gvs.shape)
        r = anp.conj(r) * fac / norm
        return r
    irfft_fun.defvjp(irfft_grad)

for rfft_fun, irfft_fun, n in ((rfft, irfft, 1), (rfftn, irfftn, None), (rfft2, irfft2, 2)):
    rfft_defvjp(rfft_fun, irfft_fun, n)

@primitive
def truncate_pad(x, shape):
    # truncate/pad x to have the appropriate shape
    slices = [slice(n) for n in shape]
    pads = list(zip(anp.zeros(len(shape), dtype=int),
               anp.maximum(0, anp.array(shape) - anp.array(x.shape))))
    return anp.pad(x, pads, 'constant')[slices]
truncate_pad.defvjp(lambda g, ans, vs, gvs, x, shape: match_complex(vs, truncate_pad(g, vs.shape)))

## TODO: could be made less stringent, to fail only when repeated axis has different values of s
def check_no_repeated_axes(*args, **kwargs):
        try:
            if len(args) == 2:
                axes = args[1]
            else:
                axes = kwargs['axes']
            axes_set = set(axes)
            if len(axes) != len(axes_set):
                raise NotImplementedError("FFT gradient for repeated axes not implemented.")
        except (TypeError, KeyError):
            # no iterable axes argument
            pass

def check_even_shape(axes, shape):
    if shape[axes[-1]] % 2 != 0:
        raise NotImplementedError("Real FFT gradient for odd lengthed last axes is not implemented.")

def find_axes(nd, n, *args, **kwargs):
    """ implement the default behavior of axis and axes parameters,
        returns a list of axes the fft is ran on. We need to probe
        into the details of rfft and irfft due to the compression
        storage.
    """
    if len(args) == 2:
        axes = args[1]
    else:
        axes = kwargs.get('axes', None)
        if axes is None: axes = kwargs.get('axis', None)

    if axes is None:
        if n == 1:
            axes = nd - 1
        elif n == 2:
            axes = [nd -2, nd -1]
        else:
            axes = list(range(nd))
    try:
        len(axes)
    except:
        axes = [axes]
    return axes

def make_rfft_factors(axes, facshape, normshape):
    """ make the compression factors and compute the normalization
        for irfft and rfft.
    """
    norm = 1.0
    for i in axes: norm = norm * normshape[i]

    # inplace modification is fine because we produce a constant
    # which doesn't go into autograd.
    # For same reason could have used numpy rather than anp.
    # but we already imported anp, so use it instead.
    fac = anp.zeros(facshape)
    fac[...] = 2
    index = [slice(None)] * len(facshape)
    index[axes[-1]] = (0, -1)
    fac[tuple(index)] = 1
    return fac, norm

