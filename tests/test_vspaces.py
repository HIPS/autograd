from functools import reduce
from autograd.core import vspace
from autograd.numpy.numpy_vspaces import ArrayVSpace
from autograd.test_util import check_grads, scalar_close
import numpy as np
import itertools as it

def check_vspace(value):
    vs = vspace(value)
    # --- required attributes ---
    size       = vs.size
    add        = vs.add
    scalar_mul = vs.scalar_mul
    inner_prod = vs.inner_prod
    randn      = vs.randn
    zeros      = vs.zeros
    ones       = vs.ones
    standard_basis = vs.standard_basis

    # --- util ---
    def randns(N=2):
        return [randn() for i in range(N)]
    def rand_scalar():
        return float(np.random.randn())
    def rand_scalars(N=2):
        return [rand_scalar() for i in range(N)]
    def vector_close(x, y):
        z = randn()
        return scalar_close(inner_prod(z, x), inner_prod(z, y))
    # --- vector space axioms ---
    def associativity_of_add(x, y, z):
        return vector_close(add(x, add(y, z)),
                            add(add(x, y), z))
    def commutativity_of_add(x, y):
        return vector_close(add(x, y), add(y, x))
    def identity_element_of_add(x):
        return vector_close(add(zeros(), x), x)
    def inverse_elements_of_add(x):
        return vector_close(zeros(), add(x, scalar_mul(x, -1.0)))
    def compatibility_of_scalar_mul_with_field_mul(x, a, b):
        return vector_close(scalar_mul(x, a * b),
                            scalar_mul(scalar_mul(x, a), b))
    def identity_element_of_scalar_mul(x):
        return vector_close(scalar_mul(x, 1.0), x)
    def distributivity_of_scalar_mul_wrt_vector_add(x, y, a):
        return vector_close(scalar_mul(add(x, y), a),
                            add(scalar_mul(x, a),
                                scalar_mul(y, a)))
    def distributivity_of_scalar_mul_wrt_scalar_add(x, a, b):
        return vector_close(scalar_mul(x, a + b),
                            add(scalar_mul(x, a),
                                scalar_mul(x, b)))
    # --- closure ---
    def add_preserves_vspace(x, y):
        return vs == vspace(add(x, y))
    def scalar_mul_preserves_vspace(x, a):
        return vs == vspace(scalar_mul(x, a))
    # --- inner product axioms ---
    def symmetry(x, y): return scalar_close(inner_prod(x, y), inner_prod(y, x))
    def linearity(x, y, a): return scalar_close(inner_prod(scalar_mul(x, a), y),
                                                a * inner_prod(x, y))
    def positive_definitive(x): return 0 < inner_prod(x, x)
    def inner_zeros(): return scalar_close(0, inner_prod(zeros(), zeros()))
    # --- basis vectors and special vectors---
    def basis_orthonormality():
        return all(
            [scalar_close(inner_prod(x, y), 1.0 * (ix == iy))
             for (ix, x), (iy, y) in it.product(enumerate(standard_basis()),
                                                enumerate(standard_basis()))])
    def ones_sum_of_basis_vects():
        return vector_close(reduce(add, standard_basis()), ones())
    def basis_correct_size():
        return len(list(standard_basis())) == size
    def basis_correct_vspace():
        return (vs == vspace(x) for x in standard_basis())
    def zeros_correct_vspace():
        return vs == vspace(zeros())
    def ones_correct_vspace():
        return vs == vspace(ones())
    def randn_correct_vspace():
        return vs == vspace(randn())

    assert associativity_of_add(*randns(3))
    assert commutativity_of_add(*randns())
    assert identity_element_of_add(randn())
    assert inverse_elements_of_add(randn())
    assert compatibility_of_scalar_mul_with_field_mul(randn(), *rand_scalars())
    assert identity_element_of_scalar_mul(randn())
    assert distributivity_of_scalar_mul_wrt_vector_add(randn(), randn(), rand_scalar())
    assert distributivity_of_scalar_mul_wrt_scalar_add(randn(), *rand_scalars())
    assert add_preserves_vspace(*randns())
    assert scalar_mul_preserves_vspace(randn(), rand_scalar())
    assert symmetry(*randns())
    assert linearity(randn(), randn(), rand_scalar())
    assert positive_definitive(randn())
    assert inner_zeros()
    assert basis_orthonormality()
    assert ones_sum_of_basis_vects()
    assert basis_correct_size()
    assert basis_correct_vspace()
    assert zeros_correct_vspace()
    assert ones_correct_vspace()
    assert randn_correct_vspace()

    # --- grads of basic operations ---
    check_grads(add)(*randns())
    check_grads(scalar_mul)(randn(), rand_scalar())
    check_grads(inner_prod)(*randns())

def test_array_vspace(): check_vspace(np.zeros((3,2)))
def test_array_vspace_0_dim(): check_vspace(0.0)
def test_array_vspace_complex(): check_vspace(1.0j*np.zeros((2,1)))
def test_list_vspace(): check_vspace([1.0, np.zeros((2,1))])
def test_tuple_vspace(): check_vspace((1.0, np.zeros((2,1))))
def test_dict_vspace(): check_vspace({'a': 1.0, 'b': np.zeros((2,1))})
def test_mixed_vspace(): check_vspace({'x' : [0.0, np.zeros((3,1))],
                                       'y' : ({'a' : 0.0}, [0.0])})
