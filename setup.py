from distutils.core import setup

setup(name='autograd',
      version='1.0',
      description='Autograd: Computes derivatives of complicated numpy code.',
      author='Dougal Maclaurin and David Duvenaud',
      author_email = "maclaurin@physics.harvard.edu, dduvenaud@seas.harvard.edu",
      packages=['autograd'],
      long_description="Stateless reverse-mode autodiff implementation that also offers higher-order derivatives.")
