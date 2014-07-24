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
        self.aspirin_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O aspirin'
        self.ibuprofen_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O ibuprofen'

        # SDF
        aspirin = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        aspirin.SetProp('_Name', 'aspirin')
        self.aspirin_sdf = Chem.MolToMolBlock(aspirin)
        _, self.sdf = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(self.sdf, 'wb') as f:
            f.write(self.aspirin_sdf)

        # Gzipped SDF
        _, self.sdf_gz = tempfile.mkstemp(suffix='.sdf.gz', dir=self.temp_dir)
        with gzip.open(self.sdf_gz, 'wb') as f:
            f.write(self.aspirin_sdf)

        # SMILES
        _, self.smi = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(self.smi, 'wb') as f:
            f.write(self.aspirin_smiles)

        # Gzipped SMILES
        _, self.smi_gz = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        with gzip.open(self.smi_gz, 'wb') as f:
            f.write(self.aspirin_smiles)

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
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(self.sdf)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_sdf_gz(self):
        """
        Read a compressed SDF file.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(self.sdf_gz)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_smi(self):
        """
        Read a SMILES file.
        """
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(self.smi)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_smi_gz(self):
        """
        Read a compressed SMILES file.
        """
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(self.smi_gz)
        assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_file_like(self):
        """
        Read from a file-like object.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        reader = serial.MolReader()
        with open(self.sdf) as f:
            mols = reader.read_mols(f, mol_format='sdf')
            assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_compressed_file_like(self):
        """
        Read from a file-like object using gzip.
        """
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        reader = serial.MolReader()
        with gzip.open(self.sdf_gz) as f:
            mols = reader.read_mols(f, mol_format='sdf')
            assert ref_mol.ToBinary() == mols.next().ToBinary()

    def test_read_multiple_smiles(self):
        """
        Read multiple SMILES.
        """
        _, filename = tempfile.mkstemp(suffix='.smi', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for smiles in [self.aspirin_smiles, self.ibuprofen_smiles]:
                f.write('{}\n'.format(smiles))
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)
        mols = list(mols)
        assert len(mols) == 2
        ref_mols = [Chem.MolFromSmiles(self.aspirin_smiles.split()[0]),
                    Chem.MolFromSmiles(self.ibuprofen_smiles.split()[0])]
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == ref_mols[i].ToBinary()

    def test_read_multiconformer(self):
        """
        Read a multiconformer SDF file with multiple molecules.
        """
        mol1 = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        mol1.SetProp('_Name', 'aspirin')
        mol1 = conformers.generate_conformers(mol1, n_conformers=3,
                                              pool_multiplier=1)
        mol2 = Chem.MolFromSmiles(self.ibuprofen_smiles.split()[0])
        mol2.SetProp('_Name', 'ibuprofen')
        mol2 = conformers.generate_conformers(mol2, n_conformers=3,
                                              pool_multiplier=1)
        ref_mols = [mol1, mol2]
        assert mol1.GetNumConformers() > 1 and mol2.GetNumConformers > 1
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        with open(filename, 'wb') as f:
            for mol in ref_mols:
                for conf in mol.GetConformers():
                    f.write(Chem.MolToMolBlock(mol, confId=conf.GetId()))
                    f.write('$$$$\n')  # add molecule delimiter
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)
        mols = list(mols)
        assert len(mols) == 2
        for i in xrange(len(mols)):
            assert mols[i].ToBinary() == ref_mols[i].ToBinary()

    def test_is_same_molecule(self):
        """
        Test MolReader.is_same_molecule.
        """
        reader = serial.MolReader()
        a = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        b = Chem.MolFromSmiles(self.ibuprofen_smiles.split()[0])
        assert reader.is_same_molecule(a, a)
        assert not reader.is_same_molecule(a, b)


class TestMolWriter(TestMolIO):
    """
    Test MolWriter.
    """
    def test_write_sdf(self):
        """
        Write an SDF file.
        """
        _, filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        writer = serial.MolWriter()
        writer.open(filename)
        writer.write([ref_mol])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)

        # compare molecules
        assert mols.next().ToBinary() == ref_mol.ToBinary()

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
        ref_mol = Chem.MolFromMolBlock(self.aspirin_sdf)
        writer = serial.MolWriter()
        writer.open(filename)
        writer.write([ref_mol])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)

        # compare molecules
        assert mols.next().ToBinary() == ref_mol.ToBinary()

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
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        writer = serial.MolWriter()
        writer.open(filename)
        writer.write([ref_mol])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)

        # compare molecules
        assert mols.next().ToBinary() == ref_mol.ToBinary()

        # compare files
        with open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_smiles.split()[0]
            except AssertionError as e:
                print data
                print self.aspirin_smiles.split()[0]
                raise e

    def test_write_smiles_gz(self):
        """
        Write a compressed SMILES file.
        """
        _, filename = tempfile.mkstemp(suffix='.smi.gz', dir=self.temp_dir)
        ref_mol = Chem.MolFromSmiles(self.aspirin_smiles.split()[0])
        writer = serial.MolWriter()
        writer.open(filename)
        writer.write([ref_mol])
        reader = serial.MolReader()
        mols = reader.read_mols_from_file(filename)

        # compare molecules
        assert mols.next().ToBinary() == ref_mol.ToBinary()

        # compare files
        with gzip.open(filename) as f:
            data = f.read()
            try:
                assert data == self.aspirin_smiles.split()[0]
            except AssertionError as e:
                print data
                print self.aspirin_smiles.split()[0]
                raise e
