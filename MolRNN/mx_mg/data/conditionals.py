from abc import ABCMeta, abstractmethod
from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np

__all__ = ['Conditional', 'Delimited', 'SparseFP', 'ScaffoldFP']


class Conditional(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Delimited(Conditional):

    def __init__(self, d='\t'):
        self.d = d

    def __call__(self, line):
        # Remove the '\n' and '\r' characters
        line = line.strip('\n').strip('\r')
        # Split the string every delimiter met.
        line = line.split(self.d)
        # The smiles string is the first element
        smiles = line[0]
        # Initialise a conditional code as an array of the measured property values
        c = np.array([float(c_i) for c_i in line[1:]], dtype=np.float32)
        return smiles, c


class SparseFP(Conditional):

    def __init__(self, fp_size=1024):
        self.fp_size = fp_size

    def __call__(self, line):
        # Remove the '\n' and '\r' characters and split the string every '\t' met.
        record = line.strip('\n').strip('\r').split('\t')
        # Initialise a conditional code as an array of False elements. Each related to a specific scaffold.
        c = np.array([False, ] * self.fp_size, dtype=bool)
        # The smiles string is the first element, while the others are the conditional codes
        smiles, on_bits = record[0], record[1:]
        # If the first element after the smiles string is not ''
        if on_bits[0] != '':
            # Save a list of all the elements, other than the smiles string, expressed as integers
            on_bits = [int(_i) for _i in on_bits]
            # Change the values of the conditional array into True in correspondence of the present scaffold.
            c[on_bits] = True
        return smiles, c


class ScaffoldFP(Conditional):

    def __init__(self, scaffolds):
        if isinstance(scaffolds, str):
            # input the directory of scaffold file
            with open(scaffolds) as f:
                scaffolds = [s.strip('\n').strip('\r') for s in f.readlines()]
        else:
            try:
                assert isinstance(scaffolds, list)
            except AssertionError:
                raise TypeError

        self.scaffolds = [Chem.MolFromSmiles(s) for s in scaffolds]
        self.scaffold_fps = [Chem.RDKFingerprint(s) for s in self.scaffolds]

    def get_on_bits(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        mol_fp = Chem.RDKFingerprint(mol)

        on_bits = []
        for i, s_fp_i in enumerate(self.scaffold_fps):
            if DataStructs.AllProbeBitsMatch(s_fp_i, mol_fp):
                if mol.HasSubstructMatch(self.scaffolds[i]):
                    on_bits.append(i)

        return on_bits

    def __call__(self, line):
        if isinstance(line, str):
            smiles = line.strip('\n').strip('\r')
        else:
            smiles = line
        c = np.array([False, ] * len(self.scaffolds), dtype=bool)
        c[self.get_on_bits(smiles)] = True
        return smiles, c
