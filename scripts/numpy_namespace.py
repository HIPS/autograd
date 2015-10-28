from __future__ import absolute_import
from __future__ import print_function
# Inspecting the numpy namespace and classifying numpy's functions
import numpy as np
from collections import defaultdict
import types
import inspect
from future.utils import iteritems

heading = lambda x : "-"*20 + str(x) + "-"*20

np_types = defaultdict(list)
for name, obj in iteritems(np.__dict__):
    np_types[type(obj)].append(name)

print("Objects in numpy namespace by type:")
for t, vals in iteritems(np_types):
    print(heading(t))
    print(vals)
print("="*80)

all_ufuncs = np_types[np.ufunc]
unary_ufuncs = []
binary_ufuncs = []
other_ufuncs = []
for ufunc in all_ufuncs:
    f = np.__dict__[ufunc]
    if f.nin == 1:
        unary_ufuncs.append(ufunc)
    elif f.nin == 2:
        binary_ufuncs.append(ufunc)
    else:
        other_ufuncs.append(ufunc)

print(heading("Unary ufuncs:"))
print(sorted(unary_ufuncs))
print(heading("Binary ufuncs:"))
print(sorted(binary_ufuncs))
if other_ufuncs:
    print(heading("Other ufuncs:"))
    print(sorted(other_ufuncs))

all_regular_funcs = np_types[types.FunctionType] + np_types[types.BuiltinFunctionType]
print(heading("Stat functions with keepdims kwarg and ndarray method"))
keepdims_funcs = []
all_other_funcs = []
for func in all_regular_funcs:
    try:
        f = np.__dict__[func]
        keepdims = "keepdims" in inspect.getargspec(f).args
        axis = "axis" in inspect.getargspec(f).args
        ndarray_method = hasattr(np.ndarray, func)
        if keepdims and axis and ndarray_method:
            keepdims_funcs.append(func)
        else:
            all_other_funcs.append(func)
    except TypeError:
        pass
print(sorted(keepdims_funcs))

print(heading("All other functions"))
print(sorted(all_other_funcs))
