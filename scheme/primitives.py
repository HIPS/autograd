import operator as op
import numpy as np

primitives = {}

# ----- Numpy primitives -----

primitives['np.dot'] = (np.dot, ["(np.dot (np.transpose (outgrad)) arg_1)",
                                 "(np.dot (np.transpose arg_0) (outgrad))"])
primitives['np.exp'] = (np.exp, ["(mul (outgrad) result)"])
primitives['np.log'] = (np.exp, ["(div (outgrad) arg_0)"])
primitives['np.sin'] = (np.sin, ["(mul (outgrad) (np.cos arg_0))"])
primitives['np.cos'] = (np.cos, ["(mul (outgrad) (neg (np.sin arg_0)))"])
primitives['np.transpose'] = (np.transpose, ["(np.transpose (outgrad))"])

# ----- Operator primitives -----

primitives['add'] = (op.add, ["(outgrad)", "(outgrad)"])
primitives['div'] = (op.div, ["", ""])
primitives['mul'] = (op.mul, ["(mul (outgrad) arg_1)", "(mul (outgrad) arg_0)"])
primitives['pow'] = (op.pow, ["(mul (mul (outgrad) arg_1) (pow arg_0 (sub arg_1 1)))", ""])
primitives['sub'] = (op.sub, ["(outgrad)", "(neg (outgrad))"])
primitives['neg'] = (op.neg, ["(neg (outgrad))"])
primitives['gt'] = (op.gt, [""])
primitives['lt'] = (op.lt, [""])
