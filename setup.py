from setuptools import setup, find_packages


def main():
    setup(
        name='rdkit-utils',
        version='0.1',
        license='3-clause BSD',
        url='https://github.com/skearnes/rdkit-utils',
        description='Utilities for working with the RDKit',
        packages=find_packages(),
    )

if __name__ == '__main__':
    main()
