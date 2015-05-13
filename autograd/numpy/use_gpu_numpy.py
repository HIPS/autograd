from __future__ import absolute_import
import os

def use_gpu_numpy():
    return os.environ.get('AUTOGRAD_USE_GPU_NUMPY', 'no') == 'yes'
