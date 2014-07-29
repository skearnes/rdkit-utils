"""
Tests for miscellaneous utilities.
"""
import cPickle
import unittest

from rdkit import Chem

from rdkit_utils import PicklableMol


class TestPicklableMol(unittest.TestCase):
    def setUp(self):
        """
        Set up for tests.
        """
        self.mol = Chem.MolFromSmiles('CC(C)(C)NC[C@@H](C1=CC(=C(C=C1)O)CO)O')
        self.mol.SetProp('_Name', 'levalbuterol')

    def test_picklable_mol(self):
        """Test PicklableMol."""
        mol = cPickle.loads(cPickle.dumps(PicklableMol(self.mol),
                                          cPickle.HIGHEST_PROTOCOL))
        assert mol.HasProp('_Name')
        assert mol.GetProp('_Name') == self.mol.GetProp('_Name')

        # make sure stereochemistry is preserved
        for a, b in zip(mol.GetAtoms(), self.mol.GetAtoms()):
            assert a.GetChiralTag() == b.GetChiralTag()
