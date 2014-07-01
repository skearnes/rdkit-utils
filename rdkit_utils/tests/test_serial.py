"""
Tests for io.py.
"""
import gzip
import os
import tempfile

from rdkit import Chem

from rdkit_utils import conformers, serial


def test_read_sdf():
    """Read SDF file."""
    _, filename = tempfile.mkstemp(suffix='.sdf')
    with open(filename, 'wb') as f:
        f.write(aspirin_sdf)
    n_atoms = Chem.MolFromMolBlock(aspirin_sdf).GetNumAtoms()
    assert serial.read_mols_from_file(filename).next().GetNumAtoms() == n_atoms
    os.remove(filename)


def test_read_sdf_gz():
    """Read compressed SDF file."""
    _, filename = tempfile.mkstemp(suffix='.sdf.gz')
    with gzip.open(filename, 'wb') as f:
        f.write(aspirin_sdf)
    n_atoms = Chem.MolFromMolBlock(aspirin_sdf).GetNumAtoms()
    assert serial.read_mols_from_file(filename).next().GetNumAtoms() == n_atoms
    os.remove(filename)


def test_read_smi():
    """Read SMILES file."""
    _, filename = tempfile.mkstemp(suffix='.smi')
    with open(filename, 'wb') as f:
        f.write(aspirin_smiles)
    n_atoms = Chem.MolFromSmiles(aspirin_smiles.split()[0]).GetNumAtoms()
    assert serial.read_mols_from_file(filename).next().GetNumAtoms() == n_atoms
    os.remove(filename)


def test_read_smi_gz():
    """Read compressed SMILES file."""
    _, filename = tempfile.mkstemp(suffix='.smi.gz')
    with gzip.open(filename, 'wb') as f:
        f.write(aspirin_smiles)
    n_atoms = Chem.MolFromSmiles(aspirin_smiles.split()[0]).GetNumAtoms()
    assert serial.read_mols_from_file(filename).next().GetNumAtoms() == n_atoms
    os.remove(filename)


def test_read_file_like():
    """Read from a file-like object."""
    _, filename = tempfile.mkstemp(suffix='.sdf')
    with open(filename, 'wb') as f:
        f.write(aspirin_sdf)
    n_atoms = Chem.MolFromMolBlock(aspirin_sdf).GetNumAtoms()
    with open(filename) as f:
        mols = serial.read_mols(f, mol_format='sdf')
        mols = [mol for mol in mols]
    assert mols[0].GetNumAtoms() == n_atoms
    os.remove(filename)


def test_read_multiconformer():
    """Read multiconformer SDF file."""
    mol = Chem.MolFromMolBlock(aspirin_sdf)
    mol = conformers.generate_conformers(mol, n_conformers=2)
    assert mol.GetNumConformers() > 1
    _, filename = tempfile.mkstemp(suffix='.sdf')
    serial.write_mols_to_file([mol], filename)
    mols = serial.read_mols_from_file(filename)
    mols = [mol for mol in mols]
    assert len(mols) == 1
    assert mols[0].GetNumConformers() == mol.GetNumConformers()
    os.remove(filename)


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
    serial.write_mols_to_file(mol, filename)
    mols = serial.read_mols_from_file(filename)
    assert mols.next().GetNumAtoms() == mol.GetNumAtoms()
    os.remove(filename)


def test_write_sdf_gz():
    """Write SDF file."""
    _, filename = tempfile.mkstemp(suffix='.sdf.gz')
    mol = Chem.MolFromSmiles(aspirin_smiles.split()[0])
    serial.write_mols_to_file(mol, filename)
    mols = serial.read_mols_from_file(filename)
    assert mols.next().GetNumAtoms() == mol.GetNumAtoms()
    os.remove(filename)

aspirin_sdf = """aspirin
     RDKit

 13 13  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  8  9  1  0
  9 10  2  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
 10  5  1  0
M  END
"""

aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
ibuprofen_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O ibuprofen'
