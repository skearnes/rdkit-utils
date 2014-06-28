"""
I/O functions: reading and writing molecules.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import gzip
import numpy as np
import warnings

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def read_mols_from_file(filename, mol_format=None, remove_salts=True):
    """
    Read molecules from a file.

    Parameters
    ----------
    filename : str
        Filename.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf' and 'smi'. If not
        provided, this method will attempt to infer it from the filename.
    remove_salts : bool, optional (default True)
        Whether to remove salts.

    Returns
    -------
    An ndarray containing Mol objects.
    """

    # determine file format
    if mol_format is None:
        if filename.endswith(('.sdf', '.sdf.gz')):
            mol_format = 'sdf'
        elif filename.endswith(('.smi', '.smi.gz', '.can', '.can.gz', '.ism',
                                '.ism.gz')):
            mol_format = 'smi'
        else:
            raise NotImplementedError('Unable to guess molecule file format.')
    if mol_format not in ['sdf', 'smi']:
        raise NotImplementedError('Unsupported molecule file format ' +
                                  '"{}".'.format(mol_format))
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename)
    mols = read_mols(f, mol_format=mol_format, remove_salts=remove_salts)
    f.close()
    return mols


def read_mols(f, mol_format, remove_salts=True):
    """
    Read molecules. Supports SDF or SMILES format.

    Molecule conformers are combined into a single molecule. Two molecules
    are considered conformers of the same molecule if they:
    * Are contiguous in the file
    * Have identical SMILES strings
    * Have identical compound names (a warning is issued if compounds lack
        names)

    Parameters
    ----------
    f : file
        File-like object.
    mol_format : str
        Molecule file format. Currently supports 'sdf' and 'smi'.
    remove_salts : bool, optional (default True)
        Whether to remove salts.

    Returns
    -------
    An ndarray containing Mol objects.
    """

    # read molecules
    mols = []
    if mol_format == 'sdf':
        for mol in Chem.ForwardSDMolSupplier(f):
            mols.append(mol)
    elif mol_format == 'smi':
        lines = [line.strip().split() for line in f.readlines()]
        for line in lines:
            if len(line) > 1:
                smiles, name = line
            else:
                smiles = line
                name = None
            mol = Chem.MolFromSmiles(smiles)
            if name is not None:
                mol.SetProp('_Name', name)
            mols.append(mol)
    else:
        raise NotImplementedError('Unrecognized mol_format "{}"'.format(
            mol_format))

    # remove salts
    if remove_salts:
        salt_remover = SaltRemover()
        mols = [salt_remover.StripMol(mol) for mol in mols]

    # combine conformers
    smiles = np.asarray([Chem.MolToSmiles(mol, isomericSmiles=True,
                                          canonical=True) for mol in mols])
    names = []
    for mol in mols:
        if mol.HasProp('_Name'):
            names.append(mol.GetProp('_Name'))
        else:
            names.append(None)
    to_combine = [[mols[0]]]
    idx = 0
    for i in xrange(1, smiles.size):
        if smiles[i] == smiles[i - 1] and names[i] == names[i - 1]:
            if names[i] is None:
                warnings.warn("Combining conformers for unnamed molecules.")
            to_combine[idx].append(mols[i])
        else:
            to_combine.append([mols[i]])
            idx += 1

    combined = []
    for mols in to_combine:
        if len(mols) == 1:
            mol, = mols
        else:
            mol = Chem.Mol(mols[0])
            for other in mols[1:]:
                for conf in other.GetConformers():
                    mol.AddConformer(conf)
        combined.append(mol)
    combined = np.asarray(combined)

    return combined


def write_mols(mols, filename):
    """
    Write SDF molecules.

    Parameters
    ----------
    mols : list
        Molecules to write.
    filename : str
        Output filename.
    """
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'wb')
    else:
        f = open(filename, 'wb')
    w = Chem.SDWriter(f)
    for mol in np.atleast_1d(mols):
        if mol.GetNumConformers():
            for conf in mol.GetConformers():
                w.write(mol, confId=conf.GetId())
        else:
            w.write(mol)
    w.close()
    f.close()
