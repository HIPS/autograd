from __future__ import absolute_import
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.core import primitive
from six.moves import zip

wrap_namespace(ffto.__dict__, globals())

# TODO: make fft gradient work for a repeated axis,
# e.g. by replacing fftn with repeated calls to 1d fft along each axis
def fft_defgrad(fft_fun):
    def fft_grad(ans, x, *args, **kwargs):
        check_no_repeated_axes(*args, **kwargs)
        return lambda g: truncate_pad(fft_fun(g, *args, **kwargs), x.shape)
    fft_fun.defgrad(fft_grad)

for fft_fun in (fft, ifft, fft2, ifft2, fftn, ifftn):
    fft_defgrad(fft_fun)

fftshift.defgrad( lambda ans, x, axes=None : lambda g : anp.conj(ifftshift(anp.conj(g), axes)))
ifftshift.defgrad(lambda ans, x, axes=None : lambda g : anp.conj(fftshift(anp.conj(g), axes)))

@primitive
def truncate_pad(x, shape):
    # truncate/pad x to have the appropriate shape
    slices = [slice(n) for n in shape]
    pads = list(zip(anp.zeros(len(shape)),
               anp.maximum(0, anp.array(shape) - anp.array(x.shape))))
    return anp.pad(x, pads, 'constant')[slices]
truncate_pad.defgrad(lambda ans, x, shape: lambda g: truncate_pad(g, x.shape))

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
