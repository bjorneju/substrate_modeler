from setuptools import setup

setup(
    name="units",
    version="0.1.0",
    description="Create pyphi substrates from the ground up",
    url="https://github.com/bjorneju/pyphi_units",
    author="Bjørn E juel",
    author_email="bjorneju@gmail.com",
    license="BSD 2-clause",
    packages=["units"],
    install_requires=[
        "tqdm",
        "numpy",
        "pandas",
        "networkx",
        "pyphi",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
