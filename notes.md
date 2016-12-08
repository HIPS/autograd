
# todo
  [ ] work out vspace type system
  [ ] write systematic tests for primitive grads using
      (correct vspaces, linearity, agree with finite differences)
  [ ] expose flatten/unflatten
  [ ] make the primitives handle any necessary casting
  [ ] work out how to handle grad_with_aux (ideally self-referentially)
  [ ] check we still raise errors like "Output type {} can't be cast to float"
  [ ] use vspace in primitive gradients (e.g. for broadcasting)
  [ ] replace safe_type with int as an acceptable scalar value (cast to float)
  [ ] use either defgrad_is_zero or nograd_primitive but not both
  [ ] figure out way to handle singleton arrays rather than casting to scalar
      (maybe treat scalars as arrays of shape ()?)
  [ ] write memory tests that check for cycles
  [ ] decide how to handle backward compatibility
  [ ] merge in changes from last 6 months of main branch commits

# changes in simplifying-core-no-tape
  * Removed tape in favor of direct toposort.
    Avoids relying on execution order for correctness.
  * Created `VSpace` objects which represent explicit vector spaces,
	and describe how to perform addition, how to
	canonically map to R^N and how to create the zero vector.
  * Removed a layer of lambda nesting in defgrads.
	Simpler and faster (but does prevent some memory optimization).
  * Removed ReverseNode which kept track of the execution trace
    in favor of the nodes themselves keeping track of their parents.
	Simpler, less mutatibility, avoids pointer cycles ruining gc.
  * Universal sparse object


# types

We have three different type systems in autograd
  * Python types (e.g. `np.ndarray`): raw Python values living
    within nodes and being passed around as function arguments
  * Node types (e.g. `ArrayNode`): needed in order to allow methods
    (e.g. `sum` or `__add__`) of numerical objects
  * VSpace types (e.g. shape (3,4) array of float 64):
    representation of vector spaces in the mathematical sense. Providing
	the zero vector, functions for addition and scalar multiplication,
	and conversion to and from R^N.

In an ideal world, these would all be the same. Next most desirable would be
to have a one-to-one correspondence between Python types and Node types,
and have VSpace types as a finer grained version of Python/Node types
(e.g. numpy array of a particular shape and floating point precision).
In practice, since Python encourages duck typing, the situation is
more complicated. For example, the type `np.float64` behaves just
like native Python `float`, and users expect to be able to use them
interchangeably, but they are different types.
So we make both types correspond to the same autograd Node type
and the same VSpace type. The relationship between the types is
defined by two functions:
  * a mapping from Python types to autograd Node types
  * a mapping from Python values to VSpace types
Luckily, the two autograd types are completely independent.


# requirements of primitive functions

* If `f(x) == y` and `vspace(z) == vspace(y)`, then
  `vspace(f.vjps[0](z)) == vspace(x)`,
* `vspace(f(x))` is the same for all `x` in a given VSpace.

# design decisions for VSpaces

Should each element of the vector space correspond to a unique Python
value?
  * pro: avoid having to check all the instantiations of the vspace
    in our tests.
  * con: we'd need a different VSpace for e.g. `float`, `np.float64` etc.
    it would probably make implementing gradients a pain, since
    we'd need to do more casting (e.g. since `sin :: float -> np.float64`,
    we'd need `sin.grads[0] :: np.float64 -> float`)

Do we make float a special case of array (with shape ())?

How do we handle float32?

How do we handle complex numbers?
