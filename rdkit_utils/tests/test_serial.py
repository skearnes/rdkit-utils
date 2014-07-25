"""
Tests for serial.py.
"""
import gzip
import shutil
import tempfile
import unittest

from rdkit import Chem

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
        self.aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
        self.aspirin = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        self.aspirin.SetProp('_Name', 'aspirin')
        self.aspirin_sdf = Chem.MolToMolBlock(self.aspirin)

        # ibuprofen
        self.ibuprofen_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O ibuprofen'
        self.ibuprofen = Chem.MolFromSmiles(self.ibuprofen_smiles.split()[0])
        self.ibuprofen.SetProp('_Name', 'ibuprofen')
        self.ibuprofen_sdf = Chem.MolToMolBlock(self.ibuprofen)

        self.ref_mols = [self.aspirin, self.ibuprofen]

        # SDF
        _, self.sdf_filename = tempfile.mkstemp(suffix='.sdf',
                                                dir=self.temp_dir)
        with open(self.sdf_filename, 'wb') as f:
            f.write(self.aspirin_sdf)

        # Gzipped SDF
        _, self.sdf_gz_filename = tempfile.mkstemp(suffix='.sdf.gz',
                                                   dir=self.temp_dir)
        with gzip.open(self.sdf_gz_filename, 'wb') as f:
            f.write(self.aspirin_sdf)

        # SMILES
        _, self.smi_filename = tempfile.mkstemp(suffix='.smi',
                                                dir=self.temp_dir)
        with open(self.smi_filename, 'wb') as f:
            f.write(self.aspirin_smiles)

        # Gzipped SMILES
        _, self.smi_gz_filename = tempfile.mkstemp(suffix='.smi.gz',
                                                   dir=self.temp_dir)
        with gzip.open(self.smi_gz_filename, 'wb') as f:
            f.write(self.aspirin_smiles)

        self.reader = serial.MolReader()

    def tearDown(self):
        """
        Clean up temporary files.
        """
        shutil.rmtree(self.temp_dir)


class TestMolReader(TestMolIO):
    """
    Test MolReader.
    """
    def test_read_sdf(self):
        """
        Read an SDF file.
        """
        mols = self.reader.read_mols_from_file(self.sdf_filename)
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_sdf_gz(self):
        """
        Read a compressed SDF file.
        """
        mols = self.reader.read_mols_from_file(self.sdf_gz_filename)
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_smi(self):
        """
        Read a SMILES file.
        """
        mols = self.reader.read_mols_from_file(self.smi_filename)
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_smi_gz(self):
        """
        Read a compressed SMILES file.
        """
        mols = self.reader.read_mols_from_file(self.smi_gz_filename)
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_file_like(self):
        """
        Read from a file-like object.
        """
        with open(self.sdf_filename) as f:
            mols = self.reader.read_mols(f, mol_format='sdf')
            assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_compressed_file_like(self):
        """
        Read from a file-like object using gzip.
        """
        with gzip.open(self.sdf_gz_filename) as f:
            mols = self.reader.read_mols(f, mol_format='sdf')
            assert self.aspirin.ToBinary() == mols.next().ToBinary()

    def test_read_multiple_sdf(self):
        """
        Read a multiple-molecule SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for sdf in [self.aspirin_sdf, self.ibuprofen_sdf]:
                f.write(sdf)
                f.write('$$$$\n')  # add molecule delimiter
        mols = self.reader.read_mols_from_file(filename)
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == self.ref_mols[i].ToBinary()

    def test_read_multiple_smiles(self):
        """
        Read a multiple-molecule SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for smiles in [self.aspirin_smiles, self.ibuprofen_smiles]:
                f.write('{}\n'.format(smiles))
        mols = self.reader.read_mols_from_file(filename)
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == self.ref_mols[i].ToBinary()

    def test_read_multiconformer(self):
        """
        Read a multiconformer SDF file containing multiple molecules.
        """
        ref_mols = []
        for mol in self.ref_mols:
            expanded = conformers.generate_conformers(mol, n_conformers=3,
                                                      pool_multiplier=1)
            assert expanded.GetNumConformers() > 1
            ref_mols.append(expanded)
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in ref_mols:
                for conf in mol.GetConformers():
                    f.write(Chem.MolToMolBlock(mol, confId=conf.GetId()))
                    f.write('$$$$\n')  # add molecule delimiter
        mols = self.reader.read_mols_from_file(filename)
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == ref_mols[i].ToBinary()

    def test_is_same_molecule(self):
        """
        Test MolReader.is_same_molecule.
        """
        assert self.reader.is_same_molecule(self.aspirin, self.aspirin)
        assert not self.reader.is_same_molecule(self.aspirin, self.ibuprofen)

    def test_sanitization(self):
        """
        Test sanitization.
        """
        reader = serial.Chem

    def test_hydrogen_treament(self):
        """
        Test hydrogen treatment.
        """
        assert False

    def test_salt_treatment(self):
        """
        Test salt treatment.
        """
        assert False


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

    def test_write_sdf(self):
        """
        Write an SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        mols = self.reader.read_mols_from_file(filename)

        # compare molecules
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_sdf + '$$$$\n'
            except AssertionError as e:
                print data
                print self.aspirin_sdf
                raise e

    def test_write_sdf_gz(self):
        """
        Write a compressed SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        mols = self.reader.read_mols_from_file(filename)

        # compare molecules
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_sdf + '$$$$\n'
            except AssertionError as e:
                print data
                print self.aspirin_sdf
                raise e

    def test_write_smiles(self):
        """
        Write a SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        mols = self.reader.read_mols_from_file(filename)

        # compare molecules
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_smiles
            except AssertionError as e:
                print data
                print self.aspirin_smiles
                raise e

    def test_write_smiles_gz(self):
        """
        Write a compressed SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        self.writer.open(filename)
        self.writer.write([self.aspirin])
        mols = self.reader.read_mols_from_file(filename)

        # compare molecules
        assert self.aspirin.ToBinary() == mols.next().ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_smiles
            except AssertionError as e:
                print data
                print self.aspirin_smiles
                raise e
