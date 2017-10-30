# Autograd v1.2 update guide

Autograd v1.2 changed the interface for defining custom vector-Jacobian
products (VJPs). Luckily the change only affects users writing custom VJPs, and
should only require minor updates to the custom VJP code.

This guide is meant to explain why we made these changes (and others) in
Autograd v1.2, and to summarize everything you need to know to update your
custom VJP code.

- [Reasoning for the changes](#reasoning-for-the-changes)
- [New defvjp interface](#new-defvjp-interface)
- [Gradient checking](#gradient-checking)

## Reasoning for the changes

Here are some of the most important reasons for this update:
1. To allow us to make Autograd faster and more memory efficient, we staged the
   VJP functions to allow more garbage collection and eliminated almost all of
   the vspace metadata checks.
1. Forward-mode now comes built-in with `make_jvp`.
1. There's now a clear extension API in `autograd.extend`, so you can write
   custom VJPs or wrap your own numerical libraries.
1. Autograd is now backend-independent, making it easy to wrap other numerical
   libraries.
1. Autograd's tracing functionality is now parameterized and easily reusable,
   and we added some new tracers for
   [computation graph visualization](https://github.com/hips/autograd/blob/master/examples/dot_graph.py)
   and
   [pure-Python constant folding](https://github.com/hips/autograd/blob/master/autograd/misc/tracers.py).
1. More exhaustive, fast reverse- and forward-mode checking with `autograd.test_util.check_grads`.
1. Expensive VJPs can share work across arguments using `defvjp_argnums`.
1. These changes enabled some internal cleanups, and more features to come!

## New defvjp interface
First, here's an example of the old way to write custom primitives and VJPs:
```python
import autograd.numpy as np
from autograd import primitive

@primitive
def func(x, y, z):
    assert z != 0
    return x * y**2

func.defvjp(lambda g, ans, vs, gvs, x, y, z: g * y**2)
func.defvjp(lambda g, ans, vs, gvs, x, y, z: 2 * g * x * y, argnum=1)
func.defvjp_is_zero(argnums=[2])
```

Here's the new way to write custom VJPs for that same primitive:
```python
import autograd.numpy as np
from autograd.extend import primitive, defvjp  # defvjp is now a function

# primitives look the same as before
@primitive
def func(x, y, z):
    assert z != 0
    return x * y**2

# but we call defvjp differently
defvjp(func,
       lambda ans, x, y, z: lambda g: g * y**2,
       lambda ans, x, y, z: lambda g: 2 * g * x * y,
       None)
```

Here's a list of the `defvjp` changes illustrated in that example:
1. `defvjp` is a function, rather than a method on the `primitive` class. (Actually, `primitive` is now just a function, and no longer a class.) As a result, `func.defvjp(...)` became `defvjp(func, ...)`.
1. VJPs are staged, so that instead of writing `lambda g, ans, vs, gvs, *args: ...` we write `lambda ans, *args: lambda g: ...`. This change enables a lot of automatic garbage collection. In the above example, if we were differentiating only with respect to `x` argument of `func`, because the VJP for `func` with respect to argument index 0 doesn't need the values of `x` or `z` from the forward pass, those values aren't stored and can instead be immediately garbage-collected.
1. There are no more `vs` and `gvs` arguments. These usually weren't used, and computing vspace metadata for every intermediate value proved to contribute significant overhead for some programs. Autograd now avoids computing vspace metadata unless necessary.
1. `defvjp` lets you define VJPs with respect to multiple arguments at once, and the argnum(s) involved are often implicit.

Here's another example, this time showing how to define VJPs with respect to
specific argnums, leaving the others undefined.
```python
# OLD way to leave some VJPs undefined
func.defvjp(lambda g, ans, vs, gvs, x, y, z, w: ..., argnum=2)
func.defvjp(lambda g, ans, vs, gvs, x, y, z, w: ..., argnum=3)

# NEW way to leave some VJPs undefined
defvjp(func,
       lambda ans, x, y, z, w: lambda g: ...,
       lambda ans, x, y, z, w: lambda g: ...,
       argnums=[2, 3])
```

## Gradient checking
Here's how to do gradient checking, whether on a composite function or on your
primitive with a custom VJP:

```python
from autograd.test_util import check_grads

# check reverse-mode to second order
check_grads(my_func, modes=['rev'], order=2)(*args_for_my_func)
```
