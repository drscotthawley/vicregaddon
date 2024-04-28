from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='vicregaddon',
    version='0.0.3',
    url = 'https://github.com/drscotthawley/vicregaddon',
    license='MIT',
    author='Scott H. Hawley',
    author_email='scott.hawley@belmont.edu', 
    description='A lightweight and modular parallel PyTorch implementation of VICReg (intended for audio, but will try to be general)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops',
    ],
)
