"""
Tests for conformers.py.
"""
import numpy as np
import unittest

from rdkit import Chem

from rdkit_utils import conformers


class TestConformerGenerator(unittest.TestCase):
    """
    Tests for ConformerGenerator.
    """
    def setUp(self):
        """
        Set up tests.
        """
        aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
        self.mol = Chem.MolFromSmiles(aspirin_smiles.split()[0])
        self.mol.SetProp('_Name', 'aspirin')
        assert self.mol.GetNumConformers() == 0
        self.engine = conformers.ConformerGenerator()

    def test_generate_conformers(self):
        """
        Generate molecule conformers using default parameters.
        """
        mol = self.engine.generate_conformers(self.mol)
        assert mol.GetNumConformers() > 0

        # check that molecule names are retained
        assert self.mol.GetProp('_Name') == mol.GetProp('_Name')

    def test_mmff94_minimization(self):
        """
        Generate conformers and minimize with MMFF94 force field.
        """
        engine = conformers.ConformerGenerator(force_field='mmff94')
        mol = engine.generate_conformers(self.mol)
        assert mol.GetNumConformers() > 0

    def test_mmff94s_minimization(self):
        """
        Generate conformers and minimize with MMFF94s force field.
        """
        engine = conformers.ConformerGenerator(force_field='mmff94s')
        mol = engine.generate_conformers(self.mol)
        assert mol.GetNumConformers() > 0

    def test_embed_molecule(self):
        """
        Test ConformerGenerator.embed_molecule.
        """
        mol = self.engine.embed_molecule(self.mol)
        assert mol.GetNumConformers() > 0

    def test_minimize_conformers(self):
        """
        Test ConformerGenerator.minimize_conformers.
        """
        mol = self.engine.embed_molecule(self.mol)
        assert mol.GetNumConformers() > 0
        start = self.engine.get_conformer_energies(mol)
        self.engine.minimize_conformers(mol)
        finish = self.engine.get_conformer_energies(mol)

        # check that all minimized energies are lower
        assert np.all(start > finish), (start, finish)

    def test_get_conformer_energies(self):
        """
        Test ConformerGenerator.get_conformer_energies.
        """
        mol = self.engine.embed_molecule(self.mol)
        assert mol.GetNumConformers() > 0
        energies = self.engine.get_conformer_energies(mol)

        # check that the number of energies matches the number of
        # conformers
        assert len(energies) == mol.GetNumConformers()

    def test_prune_conformers(self):
        """
        Test ConformerGenerator.prune_conformers.
        """
        engine = conformers.ConformerGenerator(max_conformers=10)
        mol = engine.embed_molecule(self.mol)

        # check that there is more than one conformer
        assert mol.GetNumConformers() > 1
        engine.minimize_conformers(mol)
        energies = engine.get_conformer_energies(mol)
        pruned = engine.prune_conformers(mol)
        pruned_energies = engine.get_conformer_energies(pruned)

        # check that the number of conformers has not increased
        assert pruned.GetNumConformers() <= mol.GetNumConformers()

        # check that lowest energy conformer was selected
        assert np.allclose(min(energies), min(pruned_energies))

        # check that pruned energies are taken from the original set
        for energy in pruned_energies:
            assert np.allclose(min(np.fabs(energies - energy)), 0)

        # check that conformers are in order of increasing energy
        sort = np.argsort(pruned_energies)
        assert np.array_equal(sort, np.arange(len(pruned_energies))), sort

    def test_get_conformer_rmsd(self):
        """
        Test ConformerGenerator.get_conformer_rmsd.
        """
        engine = conformers.ConformerGenerator(max_conformers=10)
        mol = engine.embed_molecule(self.mol)

        # check that there is more than one conformer
        assert mol.GetNumConformers() > 1
        rmsd = engine.get_conformer_rmsd(mol)

        # check for a valid distance matrix
        assert rmsd.shape[0] == rmsd.shape[1] == mol.GetNumConformers()
        assert np.allclose(np.diag(rmsd), 0)
        assert np.array_equal(rmsd, rmsd.T)

        # check for non-zero off-diagonal values
        assert np.all(rmsd[np.triu_indices_from(rmsd, k=1)] > 0), rmsd
