# TL;DR

## Do use
* [Most](../autograd/numpy/numpy_grads.py) of numpy's functions
* [Most](../autograd/numpy/numpy_extra.py) numpy.ndarray methods
* [Some](../autograd/scipy/scipy_grads.py) scipy functions
* Indexing and slicing of arrays `x = A[3, :, 2:4]`
* Explicit array creation from lists `A = np.array([x, y])`

## Don't use
* Assignment to arrays `A[0,0] = x`
* Implicit casting to arrays `A = np.exp([x, y])`
* `A.dot(B)` notation (use `np.dot(A, B)` instead)
