from __future__ import absolute_import
from .container_types import make_tuple, make_list, make_dict
from .convenience_wrappers import (
    make_vjp, grad, multigrad_dict, elementwise_grad, value_and_grad,
    grad_and_aux, hessian_tensor_product, hessian_vector_product, hessian,
    jacobian, tensor_jacobian_product, vector_jacobian_product, grad_named,
    checkpoint, make_hvp, make_jvp, make_ggnvp, make_jvp,
    deriv, holomorphic_grad)
from .core import primitive, notrace_primitive

isinstance = notrace_primitive(isinstance)
