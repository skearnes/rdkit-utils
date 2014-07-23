"""
Conformer generation.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


class ConformerGenerator(object):
    """
    Generate molecule conformers.

    References
    ----------
    * http://rdkit.org/docs/GettingStartedInPython.html
      #working-with-3d-molecules
    * http://pubs.acs.org/doi/full/10.1021/ci2004658

    Parameters
    ----------
    n_conformers : int, optional (default 1)
        Maximum number of conformers to generate (after pruning).
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for pruning conformers. If None or negative, no
        pruning is performed.
    force_field : str, optional (default 'uff')
        Force field to use for conformer energy minimization. Options are
        'uff', 'mmff94', and 'mmff94s'.
    prune_after_minimization : bool, optional (default True)
        Whether to prune conformers by RMSD after minimization. If False,
        pool_multiplier is set to 1.
    pool_multiplier : int, optional (default 10)
        Factor to multiply by n_conformers to generate the initial
        conformer pool. Since conformers are pruned after energy
        minimization, increasing the size of the pool increases the chance
        of identifying n_conformers unique conformers.
    """
    def __init__(self, n_conformers=1, rmsd_threshold=0.5, force_field='uff',
                 prune_after_minimization=True, pool_multiplier=10):
        self.n_conformers = n_conformers
        if rmsd_threshold is None or rmsd_threshold < 0:
            rmsd_threshold = -1.
        self.rmsd_threshold = rmsd_threshold
        self.force_field = force_field
        self.prune_after_minimization = prune_after_minimization
        if not prune_after_minimization:
            pool_multiplier = 1
        self.pool_multiplier = pool_multiplier

    def __call__(self, mol):
        """
        Generate conformers for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        return self.generate_conformers(mol)

    def generate_conformers(self, mol):
        """
        Generate conformers for a molecule.

        This function returns a copy of the original molecule with embedded
        conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        mol = self.embed_molecule(mol)
        energies = self.minimize_conformers(mol)
        if not self.prune_after_minimization:
            return mol
        mol = self.prune_conformers(mol, energies)
        return mol

    def embed_molecule(self, mol):
        """
        Generate initial conformer pool.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol, numConfs=self.n_conformers * self.pool_multiplier,
            pruneRmsThresh=self.rmsd_threshold)
        assert mol.GetNumConformers() >= 1
        return mol

    def minimize_conformers(self, mol):
        """
        Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        energies : array_like
            Minimized conformer energies.
        """
        energies = np.zeros(mol.GetNumConformers(), dtype=float)
        if self.force_field.startswith('mmff'):
            AllChem.MMFFSanitizeMolecule(mol)
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol,
                                                           self.force_field)
        for i, conf in enumerate(mol.GetConformers()):
            if self.force_field == 'uff':
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
            elif self.force_field.startswith('mmff'):
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, mmff_props, confId=conf.GetId())
            else:
                raise ValueError("Invalid force_field " +
                                 "'{}'.".format(self.force_field))
            ff.Minimize()
            energies[i] = ff.CalcEnergy()
        return energies

    def prune_conformers(self, mol, energies):
        """
        Prune conformers from a molecule using an RMSD threshold.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        energies : array_like
            Minimized conformer energies.
        """
        rmsd = self.get_conformer_rmsd(mol)
        _, discard = self.select_conformers(energies, rmsd)
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for i in discard:
            mol.RemoveConformer(conf_ids[i])
        return mol

    @staticmethod
    def get_conformer_rmsd(mol):
        """
        Calculate conformer-conformer RMSD.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()),
                        dtype=float)
        for i, ref_conf in enumerate(mol.GetConformers()):
            for j, fit_conf in enumerate(mol.GetConformers()):
                if i >= j:
                    continue
                rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                                fit_conf.GetId())
                rmsd[j, i] = rmsd[i, j]
        return rmsd

    def select_conformers(self, energies, rmsd):
        """
        Select diverse conformers starting with lowest energy.

        Parameters
        ----------
        energies : array_like
            Conformer energies.
        rmsd : ndarray
            Conformer-conformer RMSD.

        Returns
        -------
        keep : array_like
            Indices of conformers to keep.
        discard : array_like
            Indices of conformers to discard.
        """
        if self.rmsd_threshold < 0:
            return range(len(energies)), []
        if len(energies) == 1:
            return [0], []
        sort = np.argsort(energies)
        keep = [sort[0]]  # always keep lowest-energy conformer
        discard = []
        for i in sort[1:]:
            if len(keep) >= self.n_conformers:
                discard.append(i)
                continue
            this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]
            if np.all(this_rmsd >= self.rmsd_threshold):
                keep.append(i)
            else:
                discard.append(i)
        keep = np.asarray(keep, dtype=int)
        discard = np.asarray(discard, dtype=int)
        return keep, discard


def generate_conformers(mol, **kwargs):
    """
    Generate molecule conformers.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    kwargs : dict, optional
        Keyword arguments for ConformerGenerator.
    """
    engine = ConformerGenerator(**kwargs)
    return engine(mol)
