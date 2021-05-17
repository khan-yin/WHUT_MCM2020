from setuptools import setup, Extension
import numpy as np
"""
conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

conda install ipython
pip install ninja yacs cython matplotlib tqdm opencv-python
nvcc -- version
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

git clone https://github.com/cocodataset/cocoapi.git

    #To prevent installation error do the following after commiting cocooapi :
    #using file explorer  naviagate to cocoapi\PythonAPI\setup.py and change line 14 from:
    #extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    #to
    #extra_compile_args={'gcc': ['/Qstd=c99']},
    #Based on  https://github.com/cocodataset/cocoapi/issues/51

cd cocoapi/PythonAPI
"""
# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args={'gcc': ['/Qstd=c99']},
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
