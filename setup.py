from setuptools import setup, find_packages

setup(
    name='vicregaddon',
    version='0.0.1',
    author='Scott H. Hawley',
    author_email='scott.hawley@belmont.edu', 
    description='A lightweight and modular parallel PyTorch implementation of VICReg (intended for audio, but will try to be general)',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops',
    ],
)
