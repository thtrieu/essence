from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy 

ext_modules = [
    Extension(
        "conv",
        ["conv.pyx", "cconv.c"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs = [numpy.get_include()]
    )
]

setup(
	cmdclass = {'build_ext': build_ext},
    name='convolutional',
	ext_modules = cythonize(ext_modules)
)