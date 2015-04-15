from __future__ import absolute_import
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp

wrap_namespace(ffto.__dict__, globals())

fft.defgrad(    lambda ans, x, n=None, axis=-1       : lambda g : anp.conj(fft(anp.conj(g), n, axis)))
ifft.defgrad(   lambda ans, x, n=None, axis=-1       : lambda g : anp.conj(ifft(anp.conj(g), n, axis)))
fft2.defgrad(   lambda ans, x, s=None, axes=(-2, -1) : lambda g : anp.conj(fft2(anp.conj(g), s, axes)))
ifft2.defgrad(  lambda ans, x, s=None, axes=(-2, -1) : lambda g : anp.conj(ifft2(anp.conj(g), s, axes)))
fftn.defgrad(   lambda ans, x, s=None, axes=None     : lambda g : anp.conj(fftn(anp.conj(g), s, axes)))
ifftn.defgrad(  lambda ans, x, s=None, axes=None     : lambda g : anp.conj(ifftn(anp.conj(g), s, axes)))

#TODO: Support n and s arguments for all the above methods.

fftshift.defgrad( lambda ans, x, axes=None : lambda g : anp.conj(ifftshift(anp.conj(g), axes)))
ifftshift.defgrad(lambda ans, x, axes=None : lambda g : anp.conj(fftshift(anp.conj(g), axes)))
