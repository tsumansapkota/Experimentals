from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm2x2',
    ext_modules=[
        CUDAExtension('bmm2x2_cuda', [
            'bmm2x2_cuda.cpp',
            'bmm2x2_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
