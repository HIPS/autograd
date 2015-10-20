from __future__ import absolute_import
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext
from distutils.errors import CompileError
from warnings import warn

try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
    ext = '.c'
else:
    use_cython = True
    ext = '.pyx'

class build_ext(_build_ext):
    # see http://stackoverflow.com/q/19919905 for explanation
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it's in the setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

    # if optional extension modules fail to build, keep going anyway
    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to build optional extension modules')

cmdclass = {'build_ext': build_ext}

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
)
