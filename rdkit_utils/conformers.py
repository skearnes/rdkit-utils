"""
Conformer generation.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(mol, n_conformers=1, rmsd_threshold=0.5,
                        force_field='uff', pool_multiplier=10):
    """
    Generate molecule conformers. See:
    * http://rdkit.org/docs/GettingStartedInPython.html
      #working-with-3d-molecules
    * http://pubs.acs.org/doi/full/10.1021/ci2004658

    This function returns a copy of the molecule, created before adding
    hydrogens.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    n_conformers : int, optional (default 1)
        Maximum number of conformers to generate (after pruning).
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for distinguishing conformers. If None or negative,
        no pruning is performed.
    force_field : str, optional (default 'uff')
        Force field to use for conformer energy minimization. Options are
        'uff', 'mmff94', and 'mmff94s'.
    pool_multiplier : int, optional (default 10)
        Factor to multiply by n_conformers to generate the initial
        conformer pool. Since conformers are pruned after energy
        minimization, increasing the size of the pool increases the chance
        of identifying n_conformers unique conformers.
    """
    if rmsd_threshold is None or rmsd_threshold < 0:
        rmsd_threshold = -1.
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(
        mol, numConfs=n_conformers * pool_multiplier,
        pruneRmsThresh=rmsd_threshold)
    assert mol.GetNumConformers() >= 1
    cids = np.asarray(cids, dtype=int)

    # minimize conformers and get energies
    energy = np.zeros(cids.size, dtype=float)
    if force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, force_field)
    for i, cid in enumerate(cids):
        if force_field == 'uff':
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
        elif force_field.startswith('mmff'):
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, mmff_props, confId=int(cid))
        else:
            raise ValueError("Invalid force_field '{}'.".format(force_field))
        ff.Minimize()
        energy[i] = ff.CalcEnergy()

    # calculate RMSD between minimized conformers
    rmsd = np.zeros((cids.size, cids.size), dtype=float)
    for i in xrange(cids.size):
        for j in xrange(cids.size):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(mol, mol, cids[i], cids[j])
            rmsd[j, i] = rmsd[i, j]

    # discard conformers within RMSD threshold
    _, discard = choose_conformers(energy, rmsd, n_conformers, rmsd_threshold)
    for i in discard:
        mol.RemoveConformer(cids[i])
    return mol


def choose_conformers(energy, rmsd, n_conformers, rmsd_threshold=0.5):
    """
    Select diverse conformers starting with lowest energy.

    Parameters
    ----------
    energy : array_like
        Conformer energies.
    rmsd : ndarray
        Conformer-conformer RMSD values.
    n_conformers : int
        Maximum number of conformers to choose.
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for distinguishing conformers.
    """
    if rmsd_threshold is None or rmsd_threshold < 0:
        return np.arange(len(energy))
    if len(energy) == 1:
        return [0], []
    sort = np.argsort(energy)
    keep = [sort[0]]
    discard = []
    for i in sort[1:]:
        if len(keep) >= n_conformers:
            discard.append(i)
            continue
        this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]
        if np.all(this_rmsd >= rmsd_threshold):
            keep.append(i)
        else:
            discard.append(i)
    keep = np.asarray(keep, dtype=int)
    discard = np.asarray(discard, dtype=int)
    return keep, discard
