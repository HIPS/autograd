import autograd.numpy as np
from autograd import grad
from autograd.builtins import isinstance

def test_isinstance():
  def checker(ex, type_, truthval):
    assert isinstance(ex, type_) == truthval
    return 1.

  examples = [
      [list,          [[]],          [()]],
      [np.ndarray,    [np.zeros(1)], [[]]],
      [(tuple, list), [[], ()],      [np.zeros(1)]],
  ]

  for type_, positive_examples, negative_examples in examples:
    for ex in positive_examples:
      checker(ex, type_, True)
      grad(checker)(ex, type_, True)

    for ex in negative_examples:
      checker(ex, type_, False)
      grad(checker)(ex, type_, False)
