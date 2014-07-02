rdkit-utils [![Build Status](https://travis-ci.org/skearnes/rdkit-utils.svg?branch=master)](https://travis-ci.org/skearnes/rdkit-utils)
===========

Utilities for working with the [RDKit](http://www.rdkit.org/)

The utilities in this project are designed to streamline common processes in cheminformatics, saving you time and avoiding some esoteric problems.

Examples
--------

Read molecules from multiple file formats. The `MolReader` class automatically __perceives conformers__ and can optionally __remove salts__&mdash;features that are not provided by the molecule suppliers in the RDKit:

```python
from rdkit_utils import serial

# get a molecule generator
mols = serial.read_mols_from_file('molecules.sdf.gz')  # gzipped files are OK
mols = serial.read_mols_from_file('molecules.smi')  # it can read SMILES, too

# read from a file-like object
with open('molecules.sdf.gz') as f:
    mols = serial.read_mols(f, mol_format='sdf')
    ...

# object-oriented interface
reader = serial.MolReader(remove_salts=False)
with open('molecules.smi') as f:
    mols = reader.read_mols(f, mol_format='smi')
    ...
```

Generate conformers with energy minimization _prior_ to pruning:

```python
from rdkit_utils import conformers, serial

mols = serial.read_mols_from_file('molecules.sdf.gz')
expanded = conformers.generate_conformers(mols, n_conformers=10)
```
