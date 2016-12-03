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
