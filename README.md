rdkit-utils [![Build Status](https://travis-ci.org/skearnes/rdkit-utils.svg?branch=master)](https://travis-ci.org/skearnes/rdkit-utils)
===========

Utilities for working with the [RDKit](http://www.rdkit.org/)

The utilities in this project are designed to streamline common processes in cheminformatics, saving you time and avoiding some esoteric problems.

Examples
--------

Read molecules from multiple file formats. The `MolReader` class automatically __perceives conformers__ and can optionally __remove salts__&mdash;features that are not provided by the molecule suppliers in the RDKit:

```python
from rdkit_utils import serial

reader = serial.MolReader()

# get a molecule generator
mols = reader.read_mols_from_file('molecules.sdf.gz')  # gzipped files are OK
mols = reader.read_mols_from_file('molecules.smi')  # it can read SMILES, too

# read from a file-like object
with open('molecules.sdf.gz') as f:
    mols = reader.read_mols(f, mol_format='sdf')
    ...
```

Generate conformers with minimization _prior_ to pruning:

```python
from rdkit_utils import conformers, serial

reader = serial.MolReader()
mols = reader.read_mols_from_file('molecules.sdf.gz')

engine = conformers.ConformerGenerator(max_conformers=10)
expanded = []
for mol in mols:
    expanded.append(engine.generate_conformers(mol))
...
```

Additionally, the `ConformerGenerator` class starts with a pool of conformers and prunes out conformers within an RMSD threshold.
