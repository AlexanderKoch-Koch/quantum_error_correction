from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym>=0.15.4',
    'numpy>=1.18',
]


setup(
    name='qec',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
