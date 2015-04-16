"""Convenience functions built on top of `grad`."""
from autograd.core import grad, getval

def grad_and_aux(fun, argnum=0):
    """Builds a function that returns the gradient of the first output and the
    (unmodified) second output of a function that returns two outputs."""
    def grad_and_aux_fun(*args, **kwargs):
        saved_aux = []
        def return_val_save_aux(*args, **kwargs):
            val, aux = fun(*args, **kwargs)
            saved_aux.append(aux)
            return val
        gradval = grad(return_val_save_aux, argnum)(*args, **kwargs)
        return gradval, saved_aux[0]

    return grad_and_aux_fun

def value_and_grad(fun, argnum=0):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    def double_val_fun(*args, **kwargs):
        val = fun(*args, **kwargs)
        return val, getval(val)
    gradval_and_val = grad_and_aux(double_val_fun, argnum)

    def value_and_grad_fun(*args, **kwargs):
        gradval, val = gradval_and_val(*args, **kwargs)
        return val, gradval

    return value_and_grad_fun

def elementwise_grad(fun, argnum=0):
    """Like `jacobian`, but produces a function which computes just the diagonal
    of the Jacobian, and does the computation in one pass rather than in a loop.
    Note: this is only valid if the Jacobian is diagonal. Only arrays are
    currently supported."""
    return grad(lambda x : np.sum(fun(x)), argnum=argnum)

def jacobian(fun, argnum=0):
    """Returns a function that computes the Jacobian of `fun`. If the input to
    `fun` has shape (in1, in2, ...) and the output has shape (out1, out2, ...)
    then the Jacobian has shape (out1, out2, ..., in1, in2, ...).  Only arrays
    are currently supported."""
    # TODO: consider adding this to `autograd.grad`. We could avoid repeating
    # the forward pass every time.
    def jac_fun(*args, **kwargs):
        all_grads = []
        arg_in = args[argnum]
        output = fun(*args, **kwargs)
        assert isinstance(getval(arg_in), np.ndarray), "Must have array input"
        assert isinstance(getval(output), np.ndarray), "Must have array output"
        jac = np.zeros(output.shape + arg_in.shape)
        input_slice = (slice(None),) * len(arg_in.shape)
        for idxs in it.product(*map(range, output.shape)):
            scalar_fun = lambda *args, **kwargs : fun(*args, **kwargs)[idxs]
            jac[idxs + input_slice] = grad(scalar_fun, argnum=argnum)(*args, **kwargs)
        return jac
    return jac_fun
