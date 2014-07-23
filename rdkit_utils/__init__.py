"""
Miscellaneous.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from rdkit import Chem


class PicklableMol(Chem.Mol):
    """
    RDKit Mol that preserves molecule properties when pickling.

    This class is similar to the PropertyMol class in RDKit. However, this
    class also preserves calculated properties and otherwise retains the
    interface of Chem.Mol.
    """
    def __getstate__(self):
        properties = {}
        computed_properties = {}
        for prop in self.GetPropNames(includePrivate=True):
            properties[prop] = self.GetProp(prop)
        for prop in self.GetPropNames(includePrivate=True,
                                      includeComputed=True):
            if prop not in properties:
                try:
                    computed_properties[prop] = self.GetProp(prop)
                except RuntimeError:
                    pass
        mol = self.ToBinary()
        return mol, properties, computed_properties

    def __setstate__(self, state):
        mol, properties, computed = state
        self.__init__(mol)
        self.UpdatePropertyCache()  # recalculate some computed properties
        for prop, value in properties.items():
            if not self.HasProp(prop):
                self.SetProp(prop, value)
        for prop, value in computed.items():
            if not self.HasProp(prop):
                self.SetProp(prop, value, computed=True)
