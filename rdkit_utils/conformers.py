"""
Conformer generation.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(mol, n_conformers=1, rmsd_threshold=0.5):
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
        Maximum number of conformers to generate.
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for distinguishing conformers.
    """
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, n_conformers,
                                      pruneRmsThresh=rmsd_threshold)
    assert mol.GetNumConformers() >= 1
    cids = np.asarray(cids, dtype=int)

    # minimize conformers and get energies
    energy = np.zeros(cids.size, dtype=float)
    for cid in cids:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
        ff.Minimize()
        energy[cid] = ff.CalcEnergy()

    # calculate RMSD between minimized conformers
    rmsd = np.zeros((cids.size, cids.size), dtype=float)
    for i in xrange(cids.size):
        for j in xrange(cids.size):
            if i >= j:
                continue
            rmsd[i, j] = AllChem.GetBestRMS(mol, mol, cids[i], cids[j])
            rmsd[j, i] = rmsd[i, j]

    # discard conformers within RMSD threshold
    _, discard = choose_conformers(energy, rmsd, rmsd_threshold)
    for i in discard:
        mol.RemoveConformer(cids[i])
    return mol


def choose_conformers(energy, rmsd, rmsd_threshold=0.5):
    """
    Select diverse conformers starting with lowest energy.

    Parameters
    ----------
    energy : list
        Conformer energies.
    rmsd : ndarray
        Conformer-conformer RMSD values.
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for distinguishing conformers.
    """
    if len(energy) == 1:
        return [0], []
    sort = np.argsort(energy)
    keep = [sort[0]]
    discard = []
    for i in sort[1:]:
        this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]
        if np.all(this_rmsd >= rmsd_threshold):
            keep.append(i)
        else:
            discard.append(i)
    keep = np.asarray(keep, dtype=int)
    discard = np.asarray(discard, dtype=int)
    return keep, discard
