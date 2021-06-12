# Import packages
import hashlib
import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Modules.config import *
import gc
from Modules.utils import load_data


def pre_processing(filename, division=3, vocab=True):
    data_path = os.path.join('Datasets', filename)
    # Save processed data to SMILES.txt
    new = open(os.path.join(output_folder, 'smiles.txt'), "w")
    # Read in data file line by line
    for line in open(data_path, "r"):
        line = line.rjust(len(line) + 1, "G")
        new.write(line)
    # Close files
    new.close()
    # Read in processed data file
    data = open(os.path.join(output_folder, 'smiles.txt'), "r").read()
    # Create a list of the unique characters in the dataset
    chars = list(set(data))
    # Get size (in characters) of dataset
    data_size = len(data)
    # Get number of unique characters in dataset
    vocab_size = len(chars)
    # Print dataset properties
    print("Vocab size: " + str(vocab_size))
    print("Data size in characters: " + str(data_size))
    print("Characters in data: " + str(chars))
    # Create array from characters in the dataset
    values = array(chars)
    # Create unique, numerical labels for each character between 0 and n-1, where n is the number of unique characters
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print("Array of labels for each character:")
    print(integer_encoded)
    # Encode characters into a one-hot encoding, resulting in an array of size [num unique chars, num unique chars]
    one_hot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    print("Size of array of one-hot encoded characters: " + str(one_hot_encoded.shape))
    # Read in processed data file
    data = open(os.path.join(output_folder, "smiles.txt"), "r").read()
    # Create a list of the dataset
    datalist = list(data)
    # Create an array of the dataset
    data_array = array(datalist)
    # Fit one-hot encoding to data_array
    data_array = data_array.reshape(len(data_array), 1)

    # ohe_smiles = one_hot_encoder.fit_transform(data_array).astype(int)
    # print("Size of one-hot encoded array of data: " + str(ohe_smiles.shape))
    # # Save ohe_smiles as a (compressed) file
    # np.savez_compressed(os.path.join(output_folder, "ohesmiles.npz"), ohe_smiles)
    # # Create integer SMILES data
    # int_smiles = [np.where(r == 1)[0][0] for r in ohe_smiles]
    # # Save int_smiles as a (compressed) file
    # np.savez_compressed(os.path.join(output_folder, "intsmiles.npz"), int_smiles)

    len_each_division = len(data_array) // division
    ohe_smiles_filenames = []
    int_smiles_filenames = []
    one_hot_encoder = one_hot_encoder.fit(data_array)
    for i in range(division):
        segment_data_array = data_array[i * len_each_division: (i+1) * len_each_division]
        ohe_smiles = one_hot_encoder.transform(segment_data_array).astype(int)
        print("Size of one-hot encoded array of data: " + str(ohe_smiles.shape))
        # Save ohe_smiles as a (compressed) file
        np.savez_compressed(os.path.join(output_folder, f"ohesmiles_{i}.npz"), ohe_smiles)
        # Create integer SMILES data
        int_smiles = [np.where(r == 1)[0][0] for r in ohe_smiles]
        # Save int_smiles as a (compressed) file
        np.savez_compressed(os.path.join(output_folder, f"intsmiles_{i}.npz"), int_smiles)
        ohe_smiles_filenames.append(f'ohesmiles_{i}.npz')
        int_smiles_filenames.append(f"intsmiles_{i}.npz")
        del ohe_smiles, int_smiles
        gc.collect()

    if vocab:
        # Save array with SMILES character, integer encoding, and one hot encoding (vocabulary)
        values = np.reshape(values, (np.shape(values)[0], 1))
        vocab = np.concatenate((values, integer_encoded.astype(object)), axis=1)
        vocab = vocab[vocab[:, 1].argsort()]
        vocab_values = np.reshape(vocab[:, 1], (-1, 1))
        vocab_ohe = one_hot_encoder.fit_transform(vocab_values)
        vocab_encodings = np.concatenate((vocab, vocab_ohe.astype(object)), axis=1)
        print(np.shape(vocab_encodings))
        np.save(os.path.join(output_folder, "vocab.npy"), vocab_encodings)

    data, int_data = load_data("ohesmiles_0.npz", "intsmiles_0.npz")
    input_size = np.shape(data)[2]
    del data, int_data

    return ohe_smiles_filenames, int_smiles_filenames, input_size

