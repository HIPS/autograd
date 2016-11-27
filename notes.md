
# todo
  [ ] write systematic tests for primitive grads using (correct vspaces, linearity, finite differences)
  [ ] make the primitives handle any necessary casting
  [ ] work out how to handle grad_with_aux (ideally self-referentially)
  [ ] make sure we still raise errors like "Output type {} can't be cast to float"
  [ ] use vspace in primitive gradients (e.g. for broadcasting)
  [ ] replace safe_type with int as an acceptable scalar value (cast to float)
  [ ] use either defgrad_is_zero or nograd_primitive but not both
  [ ] figure out way to handle e.g. singleton arrays rather than casting to scalar
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
