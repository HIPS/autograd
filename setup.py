from __future__ import absolute_import
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.errors import CompileError
from warnings import warn
import os

# use cython if it is importable and the environment has USE_CYTHON
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = os.getenv('USE_CYTHON', False)

# subclass the build_ext command to handle numpy include and build failures
class build_ext(_build_ext):
    # see http://stackoverflow.com/q/19919905 for explanation
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

    # if optional extension modules fail to build, keep going anyway
    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to compile optional extension modules')

# list the extension files to build
extensions = [
    Extension(
        'autograd.numpy.linalg_extra', ['autograd/numpy/linalg_extra.c'],
        extra_compile_args=['-w','-Ofast']),
]

# if using cython, regenerate the extension files from the .pyx sources
if use_cython:
    from Cython.Build import cythonize
    try:
        extensions = cythonize('**/*.pyx')
    except:
        warn('Failed to generate extension module code from Cython files')

setup(
    name='autograd',
    version='1.1.1',
    description='Efficiently computes derivatives of numpy code.',
    author='Dougal Maclaurin and David Duvenaud and Matthew Johnson',
    author_email="maclaurin@physics.harvard.edu, dduvenaud@seas.harvard.edu, mattjj@csail.mit.edu",
    packages=['autograd', 'autograd.numpy', 'autograd.scipy', 'autograd.scipy.stats'],
    install_requires=['numpy>=1.9', 'future'],
    setup_requires=['numpy>=1.9'],
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
    cmdclass={'build_ext': build_ext},
)
