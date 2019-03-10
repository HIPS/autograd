useconstraint = True

import autograd
import autograd.numpy as np
from scipy import optimize

def function(x): return x[0]**2 + x[1]**2
functionjacobian = autograd.jacobian(function)
functionhvp = autograd.hessian_vector_product(function)

def constraint(x): return np.array([x[0]**2 - (x[1]-1)**2])
constraintjacobian = autograd.jacobian(constraint)
constraintlcoh = autograd.linear_combination_of_hessians(constraint)

constraint = optimize.NonlinearConstraint(constraint, 2, np.inf, constraintjacobian, constraintlcoh)

startpoint = [1, 2]

bounds = optimize.Bounds([-np.inf, -np.inf], [np.inf, np.inf])

print optimize.minimize(
  function,
  startpoint,
  method='trust-constr',
  jac=functionjacobian,
  hessp=functionhvp,
  constraints=[constraint] if useconstraint else [],
  bounds=bounds,
)
