import setuptools
from setuptools import setup

TEST_DEPS = ["pytest==5.0.1", "pytest-runner==5.1", "pytest-cov==2.7.1"]

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name='defi strategy',
    packages=setuptools.find_packages("src"),
    version='0.1.0',
    description='momentum-based-investment-strategy Defi',
    author='Mouad BOUAIDA',
    license='MIT',
    classifiers=["Programming Language :: Python :: 3.7"],
    keywords=["momentum strategy", "defi"],
    package_dir={"": "src"},
    install_requires=requirements,
    tests_require=TEST_DEPS,
    extras_require={"test": TEST_DEPS},
)

