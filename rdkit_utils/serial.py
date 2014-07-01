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


def guess_mol_format(filename):
    """
    Guess molecule file format from filename. Currently supports SDF and
    SMILES.

    Parameters
    ----------
    filename : str
        Filename.
    """
    if filename.endswith(('.sdf', '.sdf.gz')):
        mol_format = 'sdf'
    elif filename.endswith(('.smi', '.smi.gz', '.can', '.can.gz',
                            '.ism', '.ism.gz')):
        mol_format = 'smi'
    else:
        raise NotImplementedError('Unrecognized file format.')
    return mol_format


class MolReader(object):
    """
    Read molecules from files and file-like objects. Supports SDF or SMILES
    format.

    Parameters
    ----------
    remove_salts : bool, optional (default True)
        Whether to remove salts from molecules.
    """
    def __init__(self, remove_salts=True):
        self.remove_salts = remove_salts
        self.salt_remover = SaltRemover()

    def read_mols_from_file(self, filename, mol_format=None):
        """
        Read molecules from a file.

        Parameters
        ----------
        filename : str
            Filename.
        mol_format : str, optional
            Molecule file format. Currently supports 'sdf' and 'smi'. If
            not provided, this method will attempt to infer it from the
            filename.

        Returns
        -------
        A generator yielding multi-conformer Mol objects.
        """
        if mol_format is None:
            mol_format = guess_mol_format(filename)
        if filename.endswith('.gz'):
            f = gzip.open(filename)
        else:
            f = open(filename)
        for mol in self.read_mols(f, mol_format=mol_format):
            yield mol
        f.close()

    def read_mols(self, f, mol_format):
        """
        Read molecules from a file-like object.

        Molecule conformers are grouped into a single molecule. Two
        molecules are considered conformers of the same molecule if they:
        * Are contiguous in the file
        * Have identical (canonical isomeric) SMILES strings
        * Have identical compound names (a warning is issued if compounds
            lack names)

        Parameters
        ----------
        f : file
            File-like object.
        mol_format : str
            Molecule file format. Currently supports 'sdf' and 'smi'.

        Returns
        -------
        A generator yielding multi-conformer Mol objects.
        """
        source = self._read_mols(f, mol_format)
        mol = source.next()
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
        else:
            mol_name = None
        mol_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        while True:
            try:
                new = source.next()

                # on error, skip and move to the next multiconformer mol
                if new is None:
                    mol_smiles = None
                    continue
            except StopIteration:
                break
            if new.HasProp("_Name"):
                new_name = new.GetProp("_Name")
            else:
                new_name = None
            new_smiles = Chem.MolToSmiles(new, isomericSmiles=True,
                                          canonical=True)
            assert new_smiles
            if new_smiles == mol_smiles and new_name == mol_name:
                if not new_name:
                    warnings.warn("Grouping conformers of an unnamed " +
                                  "molecule.")
                assert new.GetNumConformers() == 1
                for conf in new.GetConformers():
                    mol.AddConformer(conf)
            else:
                if self.remove_salts:
                    mol = self.salt_remover.StripMol(mol)
                yield mol
                mol = new
                if mol.HasProp("_Name"):
                    mol_name = mol.GetProp("_Name")
                else:
                    mol_name = None
                mol_smiles = Chem.MolToSmiles(mol, isomericSmiles=True,
                                              canonical=True)
        if self.remove_salts:
            mol = self.salt_remover.StripMol(mol)
        yield mol

    def _read_mols(self, f, mol_format):
        """
        Read molecules from a file-like object.

        This method returns individual conformers from a file and does not
        attempt to combine them into multiconformer Mol objects.

        Parameters
        ----------
        f : file
            File-like object.
        mol_format : str
            Molecule file format. Currently supports 'sdf' and 'smi'.

        Returns
        -------
        A generator yielding single-conformer Mol objects.
        """
        if mol_format == 'sdf':
            for mol in Chem.ForwardSDMolSupplier(f):
                yield mol
        elif mol_format == 'smi':
            for line in f.readlines():
                line = line.strip().split()
                if len(line) > 1:
                    smiles, name = line
                else:
                    smiles = line
                    name = None
                mol = Chem.MolFromSmiles(smiles)
                if name is not None:
                    mol.SetProp('_Name', name)
                yield mol
        else:
            raise NotImplementedError('Unrecognized mol_format "{}"'.format(
                mol_format))


def read_mols_from_file(filename, mol_format=None, remove_salts=True):
    """
    Read molecules from a file.

    Parameters
    ----------
    filename : str
        Filename.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf' and 'smi'. If
        not provided, this method will attempt to infer it from the
        filename.
    remove_salts : bool, optional (default True)
        Whether to remove salts from molecules.
    """
    reader = MolReader(remove_salts=remove_salts)
    for mol in reader.read_mols_from_file(filename, mol_format):
        yield mol


def read_mols(f, mol_format, remove_salts=True):
    """
    Read molecules from a file-like object.

    Parameters
    ----------
    f : file
        File-like object.
    mol_format : str
        Molecule file format. Currently supports 'sdf' and 'smi'.
    remove_salts : bool, optional (default True)
        Whether to remove salts from molecules.
    """
    reader = MolReader(remove_salts=remove_salts)
    for mol in reader.read_mols(f, mol_format):
        yield mol


class MolWriter(object):
    """
    Write molecules to files or file-like objects. Supports SDF or SMILES
    format.

    Parameters
    ----------
    f : file, optional
        File-like object.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf' and 'smi'.
    """
    def __init__(self, f=None, mol_format=None):
        self.f = f
        self.mol_format = mol_format

    def __del__(self):
        self.close()

    def open(self, filename, mol_format=None):
        """
        Open output file.

        Parameters
        ----------
        filename : str
            Filename.
        mol_format : str, optional
            Molecule file format. Currently supports 'sdf' and 'smi'.
        """
        if filename.endswith('.gz'):
            self.f = gzip.open(filename, 'wb')
        else:
            self.f = open(filename, 'wb')
        if mol_format is None:
            self.mol_format = guess_mol_format(filename)
        else:
            self.mol_format = mol_format

    def close(self):
        """
        Close output file.
        """
        if self.f is not None:
            self.f.close()

    def write(self, mols):
        """
        Write molecules to a file.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        if self.mol_format == 'sdf':
            w = Chem.SDWriter(self.f)
            for mol in np.atleast_1d(mols):
                if mol.GetNumConformers():
                    for conf in mol.GetConformers():
                        w.write(mol, confId=conf.GetId())
                else:
                    w.write(mol)
            w.close()
        elif self.mol_format == 'smi':
            w = Chem.SmilesWriter(self.f)
            for mol in np.atleast_1d(mols):
                w.write(mol)
            w.close()


def write_mols_to_file(mols, filename, mol_format=None):
    """
    Write molecules to a file.

    Parameters
    ----------
    mols : iterable
        Molecules to write.
    filename : str
        Output filename.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf' and 'smi'. If
        not provided, this method will attempt to infer it from the
        filename.
    """
    writer = MolWriter()
    writer.open(filename, mol_format)
    writer.write(mols)
    writer.close()


def write_mols(mols, f, mol_format):
    """
    Write molecules to a file-like object.

    Parameters
    ----------
    mols : iterable
        Molecules to write.
    f : file
        File-like object.
    mol_format : str
        Molecule file format. Currently supports 'sdf' and 'smi'.
    """
    writer = MolWriter(f, mol_format)
    writer.write(mols)
