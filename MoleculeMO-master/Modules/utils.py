# Import packages
import time
import numpy as np
from Modules.config import *
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, MolSurf, Lipinski


def time_elapsed(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed / 3600)
    minutes = int(int(elapsed / 60) % 60)
    seconds = int(elapsed % 60)
    return hours, minutes, seconds


# Get input tensor
def input_tensors(i, batch_dim, seq_len, data, int_data):
    # Define input and target tensor sizes
    # -1 in the computation since the last character of each smiles string is not considered
    inp = torch.Tensor((seq_len - 1) * batch_dim, 1, np.shape(data)[2])
    target = torch.Tensor((seq_len - 1) * batch_dim)
    # SMILES molecules in batch
    num_chars_in_batch = seq_len * batch_dim
    first_smiles = int((seq_len * i))
    inputs = data[first_smiles: first_smiles + num_chars_in_batch, :, :]
    targets = int_data[first_smiles: first_smiles + num_chars_in_batch]
    # Index counters for input, target
    r, s = 0, 0
    for p in range(num_chars_in_batch - 1):
        # Does not include last character in SMILES in the input data
        if p % seq_len != (seq_len - 1):
            # Copy the character into the input tensor
            inp[r, :, :] = inputs[p, :, :]
            r += 1
        if p % seq_len != 0:
            # Target data (does not include first character in SMILES)
            target[s] = targets[p]
            s += 1
    return inp, target


def input_tensors_new(i, batch_dim, seq_len, data, int_data):
    # SMILES molecules in batch
    num_chars_in_batch = seq_len * batch_dim
    first_smiles = int((seq_len * i))
    inputs = data[first_smiles: first_smiles + num_chars_in_batch, :, :]
    targets = int_data[first_smiles: first_smiles + num_chars_in_batch]
    return inputs, targets


def load_data(data_file, int_data_file=None, single_input=False, folder=output_folder):
    # Load SMILES data as integer labels and as one-hot encoding
    data = np.load(os.path.join(folder, data_file))
    data = data["arr_0"]
    data = torch.from_numpy(data).view(np.shape(data)[0], 1, np.shape(data)[1])
    print("Dataset size: " + str(data.size()))
    if not single_input:
        int_data = np.load(os.path.join(folder, int_data_file))
        int_data = int_data["arr_0"]
        int_data = torch.from_numpy(int_data)
        print("Integer dataset size: " + str(int_data.size()))
    else:
        int_data = None
    return data, int_data


def load_vocab(vocab_file='vocab.npy'):
    # Load vocab dictionary as numpy object array
    vocab = np.load(os.path.join(output_folder, vocab_file), allow_pickle=True)
    print(vocab)
    print("Vocab encodings size: " + str(np.shape(vocab)))
    return vocab


# Determines if SMILES is valid or not
def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Returns True if SMILES is valid, returns False if SMILES is invalid
    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0


# Determines LogP
def log_p(smiles):
    return Crippen.MolLogP(Chem.MolFromSmiles(smiles))


# Determines molecular weight
def mol_wt(smiles):
    return Descriptors.MolWt(Chem.MolFromSmiles(smiles))


# Determines number hydrogen bond acceptors
def num_acc(smiles):
    return Lipinski.NumHAcceptors(Chem.MolFromSmiles(smiles))


# Determines number hydrogen bond donors
def num_don(smiles):
    return Lipinski.NumHDonors(Chem.MolFromSmiles(smiles))


# Determines topological polar surface area
def pol_sur(smiles):
    return MolSurf.TPSA(Chem.MolFromSmiles(smiles))


# Determines number of rotatable bonds
def rol_bon(smiles):
    return Lipinski.NumRotatableBonds(Chem.MolFromSmiles(smiles))
