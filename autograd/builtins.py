from __future__ import absolute_import
from .tracer import notrace_primitive

isinstance_ = isinstance
isinstance = notrace_primitive(isinstance)

type_ = type
type = notrace_primitive(type)
