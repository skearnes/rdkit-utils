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
    """
    def __getstate__(self):
        properties = {}
        computed_properties = {}
        for prop in self.GetPropNames(includePrivate=True):
            properties[prop] = self.GetProp(prop)
        for prop in self.GetPropNames(includePrivate=True,
                                      includeComputed=True):
            if prop not in properties:
                computed_properties[prop] = self.GetProp(prop)
        mol = self.ToBinary()
        return mol, properties, computed_properties

    def __setstate__(self, state):
        mol, properties, computed = state
        self.__init__(mol)
        for prop, value in properties.items():
            self.SetProp(prop, value)
        for prop, value in computed.items():
            self.SetProp(prop, value, computed=True)
