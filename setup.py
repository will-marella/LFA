from setuptools import setup, find_packages

setup(
    name="lfa",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        'viz': ['matplotlib>=3.3.0', 'seaborn>=0.11.0'],
        'dataframe': ['pandas>=1.3.0'],
    },
    python_requires='>=3.8',
    author="Will Cambridge",
    description="Latent Factor Allocation for Disease Topic Modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/will-marella/LFA",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 