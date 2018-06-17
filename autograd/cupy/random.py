from __future__ import absolute_import
import cupy.random as cpr
from .cupy_wrapper import wrap_namespace

wrap_namespace(cpr.__dict__, globals())
