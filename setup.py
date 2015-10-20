from __future__ import absolute_import
from distutils.core import setup
from distutils.extension import Extension
import numpy as np  # TODO http://stackoverflow.com/q/19919905

try:
    from Cython.Distutils import build_ext
except:
    use_cython = False
    cmdclass = {}
    ext = '.c'
else:
    use_cython = True
    cmdclass = {'build_ext': build_ext}
    ext = '.pyx'

extensions = [Extension('autograd.numpy.linalg_extra', ['autograd/numpy/linalg_extra' + ext])]

setup(
    name='autograd',
    version='1.0.9',
    description='Efficiently computes derivatives of numpy code.',
    author='Dougal Maclaurin and David Duvenaud',
    author_email="maclaurin@physics.harvard.edu, dduvenaud@seas.harvard.edu",
    packages=['autograd', 'autograd.numpy', 'autograd.scipy', 'autograd.scipy.stats'],
    install_requires=['numpy>=1.9', 'six'],
    keywords=['Automatic differentiation', 'backpropagation', 'gradients',
              'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy', 'Scipy'],
    url='https://github.com/HIPS/autograd',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
    ext_modules=extensions,
    cmdclass=cmdclass,
    include_dirs=[np.get_include(),],
)
