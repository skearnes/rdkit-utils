"""
I/O functions: reading and writing molecules.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import gzip
import os

from rdkit import Chem
from rdkit.Chem import AllChem
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

    # strip gzip suffix
    if filename.endswith('.gz'):
        filename = os.path.splitext(filename)[0]

    if filename.endswith('.sdf'):
        mol_format = 'sdf'
    elif filename.endswith(('.smi', '.can', '.ism')):
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
    remove_hydrogens : bool, optional (default False)
        Whether to remove hydrogens from molecules.
    remove_salts : bool, optional (default True)
        Whether to remove salts from molecules.
    """
    def __init__(self, remove_hydrogens=False, remove_salts=True):
        self.remove_hydrogens = remove_hydrogens
        self.remove_salts = remove_salts
        self.salt_remover = SaltRemover()

    def clean_mol(self, mol):
        """
        Clean a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        if self.remove_salts:
            mol = self.salt_remover.StripMol(mol)
        return mol

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
        A generator yielding multi-conformer RDKit Mol objects.
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
        * Have identical compound names (if set)

        Parameters
        ----------
        f : file
            File-like object.
        mol_format : str
            Molecule file format. Currently supports 'sdf' and 'smi'.

        Returns
        -------
        A generator yielding (possibly multi-conformer) RDKit Mol objects.
        """
        source = self._read_mols(f, mol_format)
        parent = None
        for mol in source:
            if parent is None:
                parent = mol
                continue
            if self.is_same_molecule(parent, mol):
                if mol.GetNumConformers():
                    for conf in mol.GetConformers():
                        parent.AddConformer(conf, assignId=True)
                else:
                    continue  # skip duplicate molecules without conformers
            else:
                parent = self.clean_mol(parent)
                yield parent
                parent = mol
        parent = self.clean_mol(parent)
        yield parent

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
        A generator yielding RDKit Mol objects.
        """
        if mol_format == 'sdf':
            return self._read_sdf(f)
        elif mol_format == 'smi':
            return self._read_smiles(f)
        else:
            raise NotImplementedError('Unrecognized molecule format ' +
                                      '"{}"'.format(mol_format))

    def _read_sdf(self, f):
        """
        Read SDF molecules from a file-like object.

        Parameters
        ----------
        f : file
            File-like object.
        """
        supplier = Chem.ForwardSDMolSupplier(f, removeHs=self.remove_hydrogens)
        for mol in supplier:
            yield mol

    def _read_smiles(self, f):
        """
        Read SMILES molecules from a file-like object.

        Parameters
        ----------
        f : file
            File-like object.
        """
        for line in f.readlines():
            if not line.strip():
                continue
            line = line.strip().split()
            if len(line) > 1:
                smiles, name = line
            else:
                smiles, = line
                name = None

            # hydrogens are removed by default, which triggers sanitization
            if self.remove_hydrogens:
                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                Chem.SanitizeMol(mol)

            # compute 2D coordinates
            # this is required for preservation of stereochemistry when
            # writing to SDF files or blocks
            AllChem.Compute2DCoords(mol)

            if name is not None:
                mol.SetProp('_Name', name)
            yield mol

    def is_same_molecule(self, a, b):
        """
        Test whether two molecules are conformers of the same molecule.

        Test for:
        * Identical (canonical isomeric) SMILES strings
        * Identical compound names (if set)

        Parameters
        ----------
        a, b : RDKit Mol
            Molecules to compare.
        """
        a_name = None
        if a.HasProp('_Name'):
            a_name = a.GetProp('_Name')
        b_name = None
        if b.HasProp('_Name'):
            b_name = b.GetProp('_Name')
        a_smiles = Chem.MolToSmiles(a, isomericSmiles=True, canonical=True)
        b_smiles = Chem.MolToSmiles(b, isomericSmiles=True, canonical=True)
        assert a_smiles and b_smiles
        return a_smiles == b_smiles and a_name == b_name


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
    stereo : bool, optional (default True)
        Whether to preserve stereochemistry in output.
    """
    def __init__(self, f=None, mol_format=None, stereo=True):
        self.f = f
        self.mol_format = mol_format
        self.stereo = stereo

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
        if mol_format is not None:
            self.mol_format = mol_format
        else:
            self.mol_format = guess_mol_format(filename)

    def close(self):
        """
        Close output file.
        """
        if self.f is not None:
            self.f.close()

    def write(self, mols):
        """
        Write molecules to a file-like object.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        if self.mol_format == 'sdf':
            self._write_sdf(mols)
        elif self.mol_format == 'smi':
            self._write_smiles(mols)
        self.f.flush()  # flush changes

    def _write_sdf(self, mols):
        """
        Write molecules in SDF format.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        w = Chem.SDWriter(self.f)
        for mol in mols:
            if not self.stereo:
                mol = Chem.Mol(mol)  # create a copy before changing
                Chem.RemoveStereochemistry(mol)
            if mol.GetNumConformers():
                for conf in mol.GetConformers():
                    w.write(mol, confId=conf.GetId())
            else:
                w.write(mol)
        w.close()

    def _write_smiles(self, mols):
        """
        Write molecules in SMILES format.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        for mol in mols:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=self.stereo,
                                      canonical=True)
            self.f.write(smiles)
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
                self.f.write('\t' + name)
            self.f.write('\n')
