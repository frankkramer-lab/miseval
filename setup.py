from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
   name='miseval',
   version='1.2.2',
   description='A Metric Library for Medical Image Segmentation Evaluation',
   url='https://github.com/frankkramer-lab/miseval',
   author='Dominik Müller',
   author_email='dominik.mueller@informatik.uni-augsburg.de',
   license='GPLv3',
   long_description=long_description,
   long_description_content_type="text/markdown",
   packages=find_packages(),
   python_requires='>=3.8',
   install_requires=['numpy<1.22,>=1.18',
                     'scikit-learn>=1.0.2',
                     'scikit-image>=0.19.1',
                     'scipy>=1.7.3',
                     'hausdorff>=0.2.6',
                     'numba>=0.54.0,<=0.55.2',
                     'dictances>=1.5.3'],
   classifiers=["Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",

                "Intended Audience :: Healthcare Industry",
                "Intended Audience :: Science/Research",

                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Image Recognition",
                "Topic :: Scientific/Engineering :: Medical Science Apps.",
                "Topic :: Scientific/Engineering :: Mathematics"]
)
