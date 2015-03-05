from distutils.core import setup

setup(name='FunkyYak',
      version='1.0',
      description='FunkyYak: Computes derivatives of complicated numpy code.',
      author='Dougal Maclaurin and David Duvenaud',
      author_email = "macLaurin@physics.harvard.edu, dduvenaud@seas.harvard.edu",
      packages=['funkyyak'],
      long_description="Stateless reverse-mode autodiff implementation that also offers higher-order derivatives.")
