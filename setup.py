from __future__ import absolute_import
from distutils.core import setup
setup(name='autograd',
      version='1.0.9',
      description='Efficiently computes derivatives of numpy code.',
      author='Dougal Maclaurin and David Duvenaud',
      author_email="maclaurin@physics.harvard.edu, dduvenaud@seas.harvard.edu",
      packages=['autograd', 'autograd.numpy', 'autograd.scipy', 'autograd.scipy.stats'],
      install_requires=['numpy>=1.8', 'six'],
      keywords=['Automatic differentiation', 'backpropagation', 'gradients',
                'machine learning', 'optimization', 'neural networks',
                'Python', 'Numpy', 'Scipy'],
      url='https://github.com/HIPS/autograd',
      license='MIT',
      classifiers=['Development Status :: 4 - Beta',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4'])
