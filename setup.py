#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jose Gallego-Posada",
    author_email="jgalle29@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="torch-coop is a toolkit for Lagrangian-based constrained optimization in Pytorch.",
    entry_points={
        "console_scripts": [
            "torch_coop=torch_coop.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="torch_coop",
    name="torch_coop",
    packages=find_packages(include=["torch_coop", "torch_coop.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/gallego-posada/torch_coop",
    version="0.1.0",
    zip_safe=False,
)
