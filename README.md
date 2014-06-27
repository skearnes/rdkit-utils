rdkit-utils
===========

Utilities for working with the [RDKit](http://www.rdkit.org/)

The utilities in this project are designed to streamline common processes in cheminformatics, saving you time and avoiding some esoteric problems.

Examples
--------

Read molecules from multiple file formats. This function automatically __perceives conformers__ and can optionally __remove salts__&mdash;features that are not provided by the molecule suppliers in the RDKit:

```python
from rdkit_utils import serial

mols = serial.read_mols('molecules.sdf.gz')  # gzipped files are OK
mols = serial.read_mols('molecules.smi')  # it can read SMILES, too
```

Generate conformers:

```python
from rdkit_utils import conformers, serial

mols = serial.read_mols('molecules.sdf.gz')
expanded = conformers.generate_conformers(mols, n_conformers=10)
```
