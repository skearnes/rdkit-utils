"""
Tests for conformers.py.
"""
from rdkit import Chem

from rdkit_utils import conformers


def test_generate_conformers():
    """Generate molecule conformers."""
    mol = Chem.MolFromSmiles(test_smiles.split()[0])
    assert mol.GetNumConformers() == 0
    mol = conformers.generate_conformers(mol)
    assert mol.GetNumConformers() > 0


def test_mmff94_minimization():
    """Generate conformers and minimize with MMFF94."""
    mol = Chem.MolFromSmiles(test_smiles.split()[0])
    assert mol.GetNumConformers() == 0
    mol = conformers.generate_conformers(mol, force_field='mmff94')
    assert mol.GetNumConformers() > 0


def test_mmff94s_minimization():
    """Generate conformers and minimize with MMFF94s."""
    mol = Chem.MolFromSmiles(test_smiles.split()[0])
    assert mol.GetNumConformers() == 0
    mol = conformers.generate_conformers(mol, force_field='mmff94s')
    assert mol.GetNumConformers() > 0

test_sdf = """aspirin
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

test_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
