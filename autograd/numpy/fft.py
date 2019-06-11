from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace

wrap_namespace(ffto.__dict__, globals())

# TODO: make fft gradient work for a repeated axis,
# e.g. by replacing fftn with repeated calls to 1d fft along each axis
def fft_grad(get_args, fft_fun, ans, x, *args, **kwargs):
    axes, s, norm = get_args(x, *args, **kwargs)
    check_no_repeated_axes(axes)
    vs = vspace(x)
    return lambda g: match_complex(x, truncate_pad(fft_fun(g, *args, **kwargs), vs.shape))

defvjp(fft, lambda *args, **kwargs:
        fft_grad(get_fft_args, fft, *args, **kwargs))
defvjp(ifft, lambda *args, **kwargs:
        fft_grad(get_fft_args, ifft, *args, **kwargs))

defvjp(fft2, lambda *args, **kwargs:
        fft_grad(get_fft_args, fft2, *args, **kwargs))
defvjp(ifft2, lambda *args, **kwargs:
        fft_grad(get_fft_args, ifft2, *args, **kwargs))

defvjp(fftn, lambda *args, **kwargs:
        fft_grad(get_fft_args, fftn, *args, **kwargs))
defvjp(ifftn, lambda *args, **kwargs:
        fft_grad(get_fft_args, ifftn, *args, **kwargs))

def rfft_grad(get_args, irfft_fun, ans, x, *args, **kwargs):
    axes, s, norm = get_args(x, *args, **kwargs)
    vs = vspace(x)
    gvs = vspace(ans)
    check_no_repeated_axes(axes)
    if s is None: s = [vs.shape[i] for i in axes]
    check_even_shape(s)

    # s is the full fft shape
    # gs is the compressed shape
    gs = list(s)
    gs[-1] = gs[-1] // 2  + 1
    fac = make_rfft_factors(axes, gvs.shape, gs, s, norm)
    def vjp(g):
        g = anp.conj(g / fac)
        r = match_complex(x, truncate_pad((irfft_fun(g, *args, **kwargs)), vs.shape))
        return r
    return vjp

def irfft_grad(get_args, rfft_fun, ans, x, *args, **kwargs):
    axes, gs, norm = get_args(x, *args, **kwargs)
    vs = vspace(x)
    gvs = vspace(ans)
    check_no_repeated_axes(axes)
    if gs is None: gs = [gvs.shape[i] for i in axes]
    check_even_shape(gs)

    # gs is the full fft shape
    # s is the compressed shape
    s = list(gs)
    s[-1] = s[-1] // 2 + 1
    def vjp(g):
        r = match_complex(x, truncate_pad((rfft_fun(g,  *args, **kwargs)), vs.shape))
        fac = make_rfft_factors(axes, vs.shape, s, gs, norm)
        r = anp.conj(r) * fac
        return r
    return vjp

defvjp(rfft, lambda *args, **kwargs:
        rfft_grad(get_fft_args, irfft, *args, **kwargs))

defvjp(irfft, lambda *args, **kwargs:
        irfft_grad(get_fft_args, rfft, *args, **kwargs))

defvjp(rfft2, lambda *args, **kwargs:
        rfft_grad(get_fft2_args, irfft2, *args, **kwargs))

defvjp(irfft2, lambda *args, **kwargs:
        irfft_grad(get_fft2_args, rfft2, *args, **kwargs))

defvjp(rfftn, lambda *args, **kwargs:
        rfft_grad(get_fftn_args, irfftn, *args, **kwargs))

defvjp(irfftn, lambda *args, **kwargs:
        irfft_grad(get_fftn_args, rfftn, *args, **kwargs))

defvjp(fftshift,  lambda ans, x, axes=None : lambda g:
                 match_complex(x, anp.conj(ifftshift(anp.conj(g), axes))))
defvjp(ifftshift, lambda ans, x, axes=None : lambda g:
                 match_complex(x, anp.conj(fftshift(anp.conj(g), axes))))

@primitive
def truncate_pad(x, shape):
    # truncate/pad x to have the appropriate shape
    slices = [slice(n) for n in shape]
    pads = tuple(zip(anp.zeros(len(shape), dtype=int),
               anp.maximum(0, anp.array(shape) - anp.array(x.shape))))
    return anp.pad(x, pads, 'constant')[tuple(slices)]
defvjp(truncate_pad, lambda ans, x, shape: lambda g:
       match_complex(x, truncate_pad(g, vspace(x).shape)))

## TODO: could be made less stringent, to fail only when repeated axis has different values of s
def check_no_repeated_axes(axes):
    axes_set = set(axes)
    if len(axes) != len(axes_set):
        raise NotImplementedError("FFT gradient for repeated axes not implemented.")

def check_even_shape(shape):
    if shape[-1] % 2 != 0:
        raise NotImplementedError("Real FFT gradient for odd lengthed last axes is not implemented.")

def get_fft_args(a, d=None, axis=-1, norm=None, *args, **kwargs):
    axes = [axis]
    if d is not None: d = [d]
    return axes, d, norm

def get_fft2_args(a, s=None, axes=(-2, -1), norm=None, *args, **kwargs):
    return axes, s, norm

def get_fftn_args(a, s=None, axes=None, norm=None, *args, **kwargs):
    if axes is None:
        axes = list(range(a.ndim))
    return axes, s, norm

def make_rfft_factors(axes, resshape, facshape, normshape, norm):
    """ make the compression factors and compute the normalization
        for irfft and rfft.
    """
    N = 1.0
    for n in normshape: N = N * n

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
    if norm is None:
        fac /= N
    return fac

