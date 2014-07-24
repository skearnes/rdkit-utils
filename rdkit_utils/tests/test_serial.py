"""
Tests for io.py.
"""
import gzip
import os
import shutil
import tempfile
import unittest

from rdkit import Chem

from rdkit_utils import conformers, serial


class TestMolReader(unittest.TestCase):
    """
    Tests for MolReader.
    """
    def setUp(self):
        """
        Write SDF and SMILES molecules to temporary files.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
        self.ibuprofen_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O ibuprofen'

        # SDF
        aspirin = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        aspirin.SetProp('_Name', 'aspirin')
        self.aspirin_sdf = Chem.MolToMolBlock(aspirin)
        _, self.sdf = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(self.sdf, 'wb') as f:
            f.write(self.aspirin_sdf)

        # Gzipped SDF
        _, self.sdf_gz = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        with gzip.open(self.sdf_gz, 'wb') as f:
            f.write(self.aspirin_sdf)

        # SMILES
        _, self.smi = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(self.smi, 'wb') as f:
            f.write(self.aspirin_smiles)

        # Gzipped SMILES
        _, self.smi_gz = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        with gzip.open(self.smi_gz, 'wb') as f:
            f.write(self.aspirin_smiles)

    def tearDown(self):
        """
        Clean up temporary files.
        """
        shutil.rmtree(self.temp_dir)

    def test_read_sdf(self):
        """
        Read an SDF file.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        mols = serial.read_mols_from_file(self.sdf)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_sdf_gz(self):
        """
        Read a compressed SDF file.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        mols = serial.read_mols_from_file(self.sdf_gz)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_smi(self):
        """
        Read a SMILES file.
        """
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        mols = serial.read_mols_from_file(self.smi)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_smi_gz(self):
        """
        Read a compressed SMILES file.
        """
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        mols = serial.read_mols_from_file(self.smi_gz)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_file_like(self):
        """
        Read from a file-like object.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        with open(self.sdf) as f:
            mols = serial.read_mols(f, mol_format='sdf')
            assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_compressed_file_like(self):
        """
        Read from a file-like object using gzip.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        with gzip.open(self.sdf_gz) as f:
            mols = serial.read_mols(f, mol_format='sdf')
            assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_multiconformer(self):
        """
        Read a multiconformer SDF file.
        """
        mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        mol = conformers.generate_conformers(mol, n_conformers=2)
        assert mol.GetNumConformers() > 1
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        serial.write_mols_to_file([mol], filename)
        mols = serial.read_mols_from_file(filename)
        mols = [m for m in mols]
        assert len(mols) == 1
        assert mols[0].GetNumConformers() == mol.GetNumConformers()


def test_read_multiple_smiles():
    """Read multiple SMILES file."""
    _, filename = tempfile.mkstemp(suffix='.smi')
    with open(filename, 'wb') as f:
        f.write("{}\n{}\n".format(aspirin_smiles, ibuprofen_smiles))
    mols = serial.read_mols_from_file(filename)
    mols = [mol for mol in mols]
    assert len(mols) == 2
    assert mols[0].GetNumAtoms() == Chem.MolFromSmiles(
        aspirin_smiles.split()[0]).GetNumAtoms()
    assert mols[1].GetNumAtoms() == Chem.MolFromSmiles(
        ibuprofen_smiles.split()[0]).GetNumAtoms()


def test_read_multiple_multiconformer():
    """Read multiple multiconformer SDF file."""
    mol1 = Chem.MolFromSmiles(aspirin_smiles.split()[0])
    mol1 = conformers.generate_conformers(mol1, n_conformers=2)
    mol2 = Chem.MolFromSmiles(ibuprofen_smiles.split()[0])
    mol2 = conformers.generate_conformers(mol2, n_conformers=2)
    assert mol1.GetNumConformers() > 1
    assert mol2.GetNumConformers() > 1
    _, filename = tempfile.mkstemp(suffix='.sdf')
    serial.write_mols_to_file([mol1, mol2], filename)
    mols = serial.read_mols_from_file(filename)
    mols = [mol for mol in mols]
    assert len(mols) == 2
    assert mols[0].GetNumConformers() == mol1.GetNumConformers()
    assert mols[1].GetNumConformers() == mol2.GetNumConformers()
    os.remove(filename)


def test_write_sdf():
    """Write SDF file."""
    _, filename = tempfile.mkstemp(suffix='.sdf')
    mol = Chem.MolFromSmiles(aspirin_smiles.split()[0])
    serial.write_mols_to_file([mol], filename)
    mols = serial.read_mols_from_file(filename)
    assert mols.next().GetNumAtoms() == mol.GetNumAtoms()
    os.remove(filename)


def test_write_sdf_gz():
    """Write compressed SDF file."""
    _, filename = tempfile.mkstemp(suffix='.sdf.gz')
    mol = Chem.MolFromSmiles(aspirin_smiles.split()[0])
    serial.write_mols_to_file([mol], filename)
    mols = serial.read_mols_from_file(filename)
    assert mols.next().GetNumAtoms() == mol.GetNumAtoms()
    os.remove(filename)


def test_is_same_molecule():
    """Test MolReader.is_same_molecule."""
    reader = serial.MolReader()
    a = Chem.MolFromSmiles(aspirin_smiles.split()[0])
    b = Chem.MolFromSmiles(ibuprofen_smiles.split()[0])
    assert reader.is_same_molecule(a, a)
    assert not reader.is_same_molecule(a, b)
