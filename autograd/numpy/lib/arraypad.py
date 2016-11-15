import imp

_, pathname, _ = imp.find_module("numpy")
with open(pathname + "/lib/arraypad.py", "rt") as fp:
    src = fp.read()
src = src.replace("import numpy as np", "from autograd.numpy import numpy_wrapper as np")
exec(src, globals(), locals())
