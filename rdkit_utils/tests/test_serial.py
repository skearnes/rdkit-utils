"""
Tests for serial.py.
"""
import cPickle
import gzip
import shutil
import tempfile
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit_utils import conformers, serial


class TestMolIO(unittest.TestCase):
    """
    Base test class for molecule I/O.
    """
    def setUp(self):
        """
        Write SDF and SMILES molecules to temporary files.
        """
        self.temp_dir = tempfile.mkdtemp()

        # aspirin
        self.aspirin = self._get_mol_from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O',
                                                 'aspirin')
        self.aspirin_h = Chem.AddHs(self.aspirin)
        self.aspirin_sodium = self._get_mol_from_smiles(
            'CC(=O)OC1=CC=CC=C1C(=O)[O-].[Na+]', 'aspirin sodium')

        # levalbuterol (chiral)
        self.levalbuterol = self._get_mol_from_smiles(
            'CC(C)(C)NC[C@@H](C1=CC(=C(C=C1)O)CO)O', 'levalbuterol')
        self.levalbuterol_hcl = self._get_mol_from_smiles(
            'CC(C)(C)NC[C@@H](C1=CC(=C(C=C1)O)CO)O.Cl',
            'levalbuterol hydrochloride')

        self.ref_mols = [self.aspirin, self.levalbuterol]
        self.reader = serial.MolReader(compute_2d_coords=False)

    def _get_mol_from_smiles(self, smiles, name=None):
        """
        Construct a molecule from a SMILES string.

        Molecules loaded from SMILES strings have zero conformers, but
        molecules loaded from SDF blocks are treated as 3D and have one
        conformer even if coordinates are not set. This method dumps the
        molecule to SDF and loads it again to obtain a molecule with one
        conformer.

        Parameters
        ----------
        smiles : str
            SMILES string.
        name : str, optional
            Molecule name.
        """
        mol = Chem.MolFromSmiles(smiles)
        if name is not None:
            mol.SetProp('_Name', name)
        AllChem.Compute2DCoords(mol)  # required to preserve stereo
        sdf = Chem.MolToMolBlock(mol, includeStereo=True)
        mol_with_conf = Chem.MolFromMolBlock(sdf)
        return mol_with_conf

    def tearDown(self):
        """
        Clean up temporary files.
        """
        shutil.rmtree(self.temp_dir)

    def test_guess_mol_format(self):
        """
        Test MolIO.guess_mol_format.
        """
        mol_formats = {
            'pkl': ['test.pkl', 'test.pkl.gz', 'test.test.pkl',
                    'test.test.pkl.gz'],
            'sdf': ['test.sdf', 'test.sdf.gz', 'test.test.sdf',
                    'test.test.sdf.gz'],
            'smi': ['test.smi', 'test.smi.gz', 'test.can', 'test.can.gz',
                    'test.ism', 'test.ism.gz', 'test.test.smi',
                    'test.test.smi.gz']
        }
        for mol_format in mol_formats.keys():
            for filename in mol_formats[mol_format]:
                assert self.reader.guess_mol_format(filename) == mol_format

    def test_close_context(self):
        """
        Make sure MolIO closes files it opened.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        self.reader.open(filename)
        self.reader.close()
        assert self.reader.f.closed

        # also test the context manager
        with self.reader.open(filename):
            pass
        assert self.reader.f.closed

    def test_not_close_other(self):
        """
        Make sure MolIO doesn't close files it didn't open.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename) as f:
            reader = serial.MolReader(f, mol_format='sdf')
            reader.close()
            assert not f.closed

        # also test the context manager
        with open(filename) as g:
            with serial.MolReader(g, mol_format='sdf'):
                pass
            assert not g.closed


class TestMolReader(TestMolIO):
    """
    Test MolReader.
    """
    def test_read_sdf(self):
        """
        Read an SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_sdf_gz(self):
        """
        Read a compressed SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        with gzip.open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_smi(self):
        """
        Read a SMILES file.
        """
        self.aspirin.RemoveAllConformers()  # SMILES are read without confs
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(Chem.MolToSmiles(self.aspirin))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_smi_title(self):
        """
        Read a SMILES file with molecule titles.
        """
        self.aspirin.RemoveAllConformers()  # SMILES are read without confs
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write('{}\t{}'.format(Chem.MolToSmiles(self.aspirin), 'aspirin'))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mol = mols.next()
        assert mol.ToBinary() == self.aspirin.ToBinary()
        assert mol.GetProp('_Name') == self.aspirin.GetProp('_Name')

    def test_read_smi_gz(self):
        """
        Read a compressed SMILES file.
        """
        self.aspirin.RemoveAllConformers()  # SMILES are read without confs
        _, filename = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        with gzip.open(filename, 'wb') as f:
            f.write(Chem.MolToSmiles(self.aspirin))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_pickle(self):
        """
        Read from a pickle.
        """
        _, filename = tempfile.mkstemp(suffix='.pkl', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            cPickle.dump([self.aspirin], f, cPickle.HIGHEST_PROTOCOL)
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_pickle_gz(self):
        """
        Read from a compressed pickle.
        """
        _, filename = tempfile.mkstemp(suffix='.pkl.gz', dir=self.temp_dir)
        with gzip.open(filename, 'wb') as f:
            cPickle.dump([self.aspirin], f, cPickle.HIGHEST_PROTOCOL)
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_file_like(self):
        """
        Read from a file-like object.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin))
        with open(filename) as f:
            reader = serial.MolReader(f, mol_format='sdf')
            mols = reader.get_mols()
            assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_compressed_file_like(self):
        """
        Read from a file-like object using gzip.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        with gzip.open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin))
        with gzip.open(filename) as f:
            reader = serial.MolReader(f, mol_format='sdf')
            mols = reader.get_mols()
            assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_read_multiple_sdf(self):
        """
        Read a multiple-molecule SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in self.ref_mols:
                sdf = Chem.MolToMolBlock(mol)
                f.write(sdf)
                f.write('$$$$\n')  # add molecule delimiter
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == self.ref_mols[i].ToBinary()

    def test_read_multiple_smiles(self):
        """
        Read a multiple-molecule SMILES file.
        """
        ref_mols = []
        for mol in self.ref_mols:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            ref_mols.append(mol)
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in self.ref_mols:
                smiles = Chem.MolToSmiles(mol)
                name = mol.GetProp('_Name')
                f.write('{}\t{}\n'.format(smiles, name))
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == ref_mols[i].ToBinary()

    def test_read_multiconformer(self):
        """
        Read a multiconformer SDF file.
        """

        # generate conformers
        engine = conformers.ConformerGenerator(max_conformers=3,
                                               pool_multiplier=1)
        ref_mol = engine.generate_conformers(self.aspirin)
        assert ref_mol.GetNumConformers() > 1

        # write to disk
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for conf in ref_mol.GetConformers():
                f.write(Chem.MolToMolBlock(ref_mol, confId=conf.GetId()))
                f.write('$$$$\n')  # add molecule delimiter

        # compare
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mols = list(mols)
        assert len(mols) == 1
        # FIXME get ToBinary test to work
        # assert mols[0].ToBinary() == ref_mol.ToBinary()
        assert Chem.MolToMolBlock(mols[0]) == Chem.MolToMolBlock(ref_mol)

    def test_read_multiple_multiconformer(self):
        """
        Read a multiconformer SDF file containing multiple molecules.
        """

        # generate conformers
        ref_mols = []
        engine = conformers.ConformerGenerator(max_conformers=3,
                                               pool_multiplier=1)
        for mol in self.ref_mols:
            expanded = engine.generate_conformers(mol)
            assert expanded.GetNumConformers() > 1
            ref_mols.append(expanded)

        # write to disk
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in ref_mols:
                for conf in mol.GetConformers():
                    f.write(Chem.MolToMolBlock(mol, includeStereo=1,
                                               confId=conf.GetId()))
                    f.write('$$$$\n')  # add molecule delimiter

        # compare
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mols = list(mols)
        assert len(mols) == 2
        for mol, ref_mol in zip(mols, ref_mols):
            # FIXME get ToBinary test to work
            # assert mol.ToBinary() == ref_mol.ToBinary()
            assert Chem.MolToMolBlock(
                mol, includeStereo=1) == Chem.MolToMolBlock(
                    ref_mol, includeStereo=1)

    def test_are_same_molecule(self):
        """
        Test MolReader.are_same_molecule.
        """
        assert self.reader.are_same_molecule(self.aspirin, self.aspirin)
        assert not self.reader.are_same_molecule(self.aspirin,
                                                 self.levalbuterol)

    def test_no_remove_hydrogens(self):
        """
        Test hydrogen retention.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin_h))
        reader = serial.MolReader(remove_hydrogens=False)
        reader.open(filename)
        mols = reader.get_mols()
        # FIXME get ToBinary test to work
        # assert mols.next().ToBinary() == self.aspirin_h.ToBinary()
        assert Chem.MolToMolBlock(mols.next()) == Chem.MolToMolBlock(
            self.aspirin_h)

    def test_remove_hydrogens(self):
        """
        Test hydrogen removal.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(Chem.MolToMolBlock(self.aspirin_h))
        reader = serial.MolReader(remove_hydrogens=True)
        reader.open(filename)
        mols = reader.get_mols()
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

    def test_remove_salts(self):
        """
        Test salt removal.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in [self.aspirin_sodium, self.levalbuterol_hcl]:
                f.write(Chem.MolToMolBlock(mol))
                f.write('$$$$\n')  # molecule delimiter
        ref_mols = [self.aspirin_sodium, self.levalbuterol_hcl]
        reader = serial.MolReader(remove_salts=True)
        reader.open(filename)
        mols = reader.get_mols()
        mols = list(mols)
        assert len(mols) == 2
        for mol, ref_mol in zip(mols, ref_mols):
            assert mol.GetNumAtoms() < ref_mol.GetNumAtoms()
            desalted = self.reader.clean_mol(ref_mol)
            assert mol.ToBinary() == desalted.ToBinary()

    def test_no_remove_salts(self):
        """
        Test salt retention.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in [self.aspirin_sodium, self.levalbuterol_hcl]:
                f.write(Chem.MolToMolBlock(mol))
                f.write('$$$$\n')  # molecule delimiter
        ref_mols = [self.aspirin_sodium, self.levalbuterol_hcl]
        reader = serial.MolReader(remove_salts=False)
        reader.open(filename)
        mols = reader.get_mols()
        mols = list(mols)
        assert len(mols) == 2
        for mol, ref_mol in zip(mols, ref_mols):
            assert mol.ToBinary() == ref_mol.ToBinary()
            desalted = self.reader.clean_mol(ref_mol)
            assert mol.GetNumAtoms() > desalted.GetNumAtoms()

    def test_iterator(self):
        """
        Test MolWriter.__iter__.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in self.ref_mols:
                f.write(Chem.MolToMolBlock(mol))
                f.write('$$$$\n')  # molecule delimiter
        self.reader.open(filename)
        for i, mol in enumerate(self.reader):
            assert mol.ToBinary() == self.ref_mols[i].ToBinary()

    def test_context_manager(self):
        """
        Test using 'with' statement to read molecules.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in self.ref_mols:
                f.write(Chem.MolToMolBlock(mol))
                f.write('$$$$\n')  # molecule delimiter
        with self.reader.open(filename) as reader:
            for i, mol in enumerate(reader):
                assert mol.ToBinary() == self.ref_mols[i].ToBinary()


class TestMolWriter(TestMolIO):
    """
    Test MolWriter.
    """
    def setUp(self):
        """
        Add writer to inherited setup.
        """
        super(TestMolWriter, self).setUp()
        self.writer = serial.MolWriter()
        self.aspirin_sdf = Chem.MolToMolBlock(self.aspirin)
        self.aspirin_smiles = Chem.MolToSmiles(self.aspirin) + '\taspirin'

    def test_write_sdf(self):
        """
        Write an SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            assert data == self.aspirin_sdf + '$$$$\n'

    def test_write_sdf_gz(self):
        """
        Write a compressed SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            assert data == self.aspirin_sdf + '$$$$\n'

    def test_write_smiles(self):
        """
        Write a SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        self.aspirin.RemoveAllConformers()  # SMILES are read without confs
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            assert data.strip() == self.aspirin_smiles

    def test_write_smiles_gz(self):
        """
        Write a compressed SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        self.aspirin.RemoveAllConformers()  # SMILES are read without confs
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            assert data.strip() == self.aspirin_smiles

    def test_write_pickle(self):
        """
        Write a pickle.
        """
        _, filename = tempfile.mkstemp(suffix='.pkl', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            assert data == cPickle.dumps([self.aspirin],
                                         cPickle.HIGHEST_PROTOCOL)

    def test_write_pickle_gz(self):
        """
        Write a compressed pickle.
        """
        _, filename = tempfile.mkstemp(suffix='.pkl.gz', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        self.writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            assert data == cPickle.dumps([self.aspirin],
                                         cPickle.HIGHEST_PROTOCOL)

    def test_stereo_setup(self):
        """
        Make sure chiral reference molecule is correct.
        """
        smiles = Chem.MolToSmiles(self.levalbuterol, isomericSmiles=True)
        assert '@' in smiles  # check for stereochemistry flag

        # check that removing stereochemistry changes the molecule
        original = self.levalbuterol.ToBinary()
        AllChem.RemoveStereochemistry(self.levalbuterol)
        assert self.levalbuterol.ToBinary() != original

    def test_stereo_sdf(self):
        """
        Test stereochemistry preservation when writing to SDF.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        writer = serial.MolWriter(stereo=True)
        writer.open(filename)
        writer.write([self.levalbuterol])
        writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == self.levalbuterol.ToBinary()

    def test_stereo_smi(self):
        """
        Test stereochemistry preservation when writing to SMILES.
        """
        # FIXME avoid this and use self.levalbuterol.RemoveAllConformers()
        ref_mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.levalbuterol,
                                                      isomericSmiles=True))
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        writer = serial.MolWriter(stereo=True)
        writer.open(filename)
        writer.write([self.levalbuterol])
        writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()
        assert mols.next().ToBinary() == ref_mol.ToBinary()

    def test_no_stereo_sdf(self):
        """
        Test stereochemistry removal when writing to SDF.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        writer = serial.MolWriter(stereo=False)
        writer.open(filename)
        writer.write([self.levalbuterol])
        writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mol = mols.next()

        # make sure the written molecule differs from the reference
        assert mol.ToBinary() != self.levalbuterol.ToBinary()

        # check again after removing stereochemistry
        AllChem.RemoveStereochemistry(self.levalbuterol)
        # FIXME get ToBinary test to work
        # assert mol.ToBinary() == self.levalbuterol.ToBinary()
        assert Chem.MolToMolBlock(
            mol, includeStereo=True) == Chem.MolToMolBlock(
                self.levalbuterol, includeStereo=True)

    def test_no_stereo_smiles(self):
        """
        Test stereochemistry removal when writing to SMILES.
        """
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        writer = serial.MolWriter(stereo=False)
        writer.open(filename)
        writer.write([self.levalbuterol])
        writer.close()
        self.reader.open(filename)
        mols = self.reader.get_mols()
        mol = mols.next()

        # make sure the written molecule differs from the reference
        assert mol.ToBinary() != self.levalbuterol.ToBinary()

        # check again after removing stereochemistry
        AllChem.RemoveStereochemistry(self.levalbuterol)
        # FIXME get ToBinary test to work
        # assert mol.ToBinary() == self.levalbuterol.ToBinary()
        assert Chem.MolToSmiles(mol, isomericSmiles=True) == Chem.MolToSmiles(
            self.levalbuterol, isomericSmiles=True)

    def test_context_manager(self):
        """
        Test use of 'with' statement to write molecules.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with self.writer.open(filename) as writer:
            writer.write([self.aspirin])
        self.reader.open(filename)
        mols = self.reader.get_mols()

        # compare molecules
        assert mols.next().ToBinary() == self.aspirin.ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            assert data == self.aspirin_sdf + '$$$$\n'
