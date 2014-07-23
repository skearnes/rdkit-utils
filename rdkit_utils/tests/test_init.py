"""
Tests for miscellaneous utilities.
"""
import cPickle
from StringIO import StringIO

from rdkit_utils import serial


def test_picklable_mol():
    """Test PicklableMol."""
    mols = serial.read_mols(StringIO(test_smiles), mol_format='smi')
    mols = list(mols)
    mol = mols[0]
    assert mol.HasProp('_Name')
    state = cPickle.dumps(mol, cPickle.HIGHEST_PROTOCOL)
    new = cPickle.loads(state)
    assert new.HasProp('_Name')
    assert new.GetProp('_Name') == mol.GetProp('_Name')

    # make sure stereochemistry is preserved
    for a, b in zip(mol.GetAtoms(), new.GetAtoms()):
        assert a.GetChiralTag() == b.GetChiralTag()

test_smiles = 'CC(C)(C)NC[C@@H](C1=CC(=C(C=C1)O)CO)O levalbuterol'
