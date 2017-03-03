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
def fft_grad(get_axes, fft_fun, g, ans, vs, gvs, x, *args, **kwargs):
    axes, s = get_axes(x, *args, **kwargs)
    check_no_repeated_axes(axes)
    return match_complex(vs, truncate_pad(fft_fun(g, *args, **kwargs), vs.shape))

fft.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, fft, *args, **kwargs))
ifft.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, ifft, *args, **kwargs))

fft2.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, fft2, *args, **kwargs))
ifft2.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, ifft2, *args, **kwargs))

fftn.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, fftn, *args, **kwargs))
ifftn.defvjp(lambda *args, **kwargs:
        fft_grad(get_fft_axes, ifftn, *args, **kwargs))

fftshift.defvjp( lambda g, ans, vs, gvs, x, axes=None : match_complex(vs, anp.conj(ifftshift(anp.conj(g), axes))))
ifftshift.defvjp(lambda g, ans, vs, gvs, x, axes=None : match_complex(vs, anp.conj(fftshift(anp.conj(g), axes))))

def rfft_grad(get_axes, irfft_fun, g, ans, vs, gvs, x, *args, **kwargs):
    axes, s = get_axes(x, *args, **kwargs)

    check_no_repeated_axes(axes)
    if s is None: s = [vs.shape[i] for i in axes]
    check_even_shape(s)

    # s is the full fft shape
    # gs is the compressed shape
    gs = list(s)
    gs[-1] = gs[-1] // 2  + 1
    fac, norm = make_rfft_factors(axes, gvs.shape, gs, s)
    g = anp.conj(g / fac)
    r = match_complex(vs, truncate_pad((irfft_fun(g, *args, **kwargs)), vs.shape))
    return r * norm

    rfft_fun.defvjp(rfft_grad)

def irfft_grad(get_axes, rfft_fun, g, ans, vs, gvs, x, *args, **kwargs):
    axes, gs = get_axes(x, *args, **kwargs)

    check_no_repeated_axes(axes)
    if gs is None: gs = [gvs.shape[i] for i in axes]
    check_even_shape(gs)

    # gs is the full fft shape
    # s is the compressed shape
    s = list(gs)
    s[-1] = s[-1] // 2 + 1
    r = match_complex(vs, truncate_pad((rfft_fun(g,  *args, **kwargs)), vs.shape))
    fac, norm = make_rfft_factors(axes, vs.shape, s, gs)
    r = anp.conj(r) * fac / norm
    return r

rfft.defvjp(lambda *args, **kwargs:
        rfft_grad(get_fft_axes, irfft, *args, **kwargs))

irfft.defvjp(lambda *args, **kwargs:
        irfft_grad(get_fft_axes, rfft, *args, **kwargs))

rfft2.defvjp(lambda *args, **kwargs:
        rfft_grad(get_fft2_axes, irfft2, *args, **kwargs))

irfft2.defvjp(lambda *args, **kwargs:
        irfft_grad(get_fft2_axes, rfft2, *args, **kwargs))

rfftn.defvjp(lambda *args, **kwargs:
        rfft_grad(get_fftn_axes, irfftn, *args, **kwargs))

irfftn.defvjp(lambda *args, **kwargs:
        irfft_grad(get_fftn_axes, rfftn, *args, **kwargs))

@primitive
def truncate_pad(x, shape):
    # truncate/pad x to have the appropriate shape
    slices = [slice(n) for n in shape]
    pads = list(zip(anp.zeros(len(shape), dtype=int),
               anp.maximum(0, anp.array(shape) - anp.array(x.shape))))
    return anp.pad(x, pads, 'constant')[slices]
truncate_pad.defvjp(lambda g, ans, vs, gvs, x, shape: match_complex(vs, truncate_pad(g, vs.shape)))

## TODO: could be made less stringent, to fail only when repeated axis has different values of s
def check_no_repeated_axes(axes):
    axes_set = set(axes)
    if len(axes) != len(axes_set):
        raise NotImplementedError("FFT gradient for repeated axes not implemented.")

def check_even_shape(shape):
    if shape[-1] % 2 != 0:
        raise NotImplementedError("Real FFT gradient for odd lengthed last axes is not implemented.")

def get_fft_axes(a, d=None, axis=-1, *args, **kwargs):
    axes = [axis]
    if d is not None: d = [d]
    return axes, d

def get_fft2_axes(a, s=None, axes=(-2, -1), *args, **kwargs):
    return axes, s

def get_fftn_axes(a, s=None, axes=None, *args, **kwargs):
    if axes is None:
        axes = list(range(a.ndim))
    return axes, s

def make_rfft_factors(axes, resshape, facshape, normshape):
    """ make the compression factors and compute the normalization
        for irfft and rfft.
    """
    norm = 1.0
    for n in normshape: norm = norm * n

    # inplace modification is fine because we produce a constant
    # which doesn't go into autograd.
    # For same reason could have used numpy rather than anp.
    # but we already imported anp, so use it instead.
    fac = anp.zeros(resshape)
    fac[...] = 2
    index = [slice(None)] * len(resshape)
    if facshape[-1] <= resshape[axes[-1]]:
        index[axes[-1]] = (0, facshape[-1] - 1)
    else:
        index[axes[-1]] = (0,)
    fac[tuple(index)] = 1
    return fac, norm

