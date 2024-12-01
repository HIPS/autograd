from autograd.core import primitive_with_deprecation_warnings as primitive

from .builtins import dict, isinstance, list, tuple, type
from .differential_operators import (
    checkpoint,
    deriv,
    elementwise_grad,
    grad,
    grad_and_aux,
    grad_named,
    hessian,
    hessian_tensor_product,
    hessian_vector_product,
    holomorphic_grad,
    jacobian,
    make_ggnvp,
    make_hvp,
    make_jvp,
    make_vjp,
    multigrad_dict,
    tensor_jacobian_product,
    value_and_grad,
    vector_jacobian_product,
)
