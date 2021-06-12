# Import packages
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import random
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Modules.config import *
from Modules.utils import is_valid, log_p, mol_wt, num_acc, num_don, rol_bon


def post_processing(num_random_molecules=1000):
    # Number of characters and molecules generated
    num_characters, num_generated_molecules = 0, 0
    # Number of unique generated SMILES and that are not in the training data
    num_unq_molecules, num_novel_molecules = 0, 0
    # Number of valid molecules that aren't in the training data generated
    num_valid = 0
    # List of smiles in file, to make sure smiles are unique
    smiles_list = []
    # Test how many molecules are valid (without considering uniqueness, novelty)
    total, number_valid = 0, 0
    # Read in data file line by line
    for line in open(os.path.join(results_folder, "gen.txt"), "r"):
        total += 1
        # Ensure smiles are valid
        if is_valid(line):
            # Increment number of valid molecules generated
            number_valid += 1
    print(f"Percentage valid molecules: {number_valid / total * 100:.2f}%")
    # Training data
    training_data = list(open(os.path.join(output_folder, "smiles.txt"), "r"))
    # File with unique novel and valid generated SMILES that aren't in the training data
    generated_molecules = open(os.path.join(results_folder, "genmols.txt"), "w")
    # Read in data file line by line
    for line in open(os.path.join(results_folder, "gen.txt"), "r"):
        # Ensure molecules are unique
        if line not in smiles_list:
            smiles_list.append(line)
            num_unq_molecules += 1
            # Ensure smiles aren't in training data
            if line not in training_data:
                # Remove \n character, remove G character
                smiles = line.replace("\n", "").replace("G", "")
                # Increment number of novel molecules generated
                num_novel_molecules += 1
                # Ensure smiles are valid
                if is_valid(smiles):
                    # Copy over SMILES satisfying requirements
                    generated_molecules.write(smiles + "\n")
                    generated_molecules.flush()
                    # Increment number of valid molecules generated
                    num_valid += 1
        # Increment total number of molecules generated
        num_generated_molecules += 1
        # Add length of line to total number of characters
        num_characters += len(line)
    generated_molecules.close()
    print("Number of characters generated: " + str(num_characters))
    print("Number of molecules generated: " + str(num_generated_molecules))
    print("Number of unique molecules generated: " + str(num_unq_molecules))
    print("Number of novel and unique molecules generated: " + str(num_novel_molecules))
    print("Number of novel, unique, and valid molecules generated: " + str(num_valid))
    # List of Morgan fingerprints of molecules
    fingerprints = []
    # Read in data file line by line
    for line in open(os.path.join(results_folder, "genmols.txt"), "r"):
        line = line.replace("G", "")
        # Convert SMILES string to Morgan fingerprint
        mol = Chem.MolFromSmiles(line.replace("\n", ""))
        fingerprint = AllChem.GetMorganFingerprint(mol, 2)
        # Add to list of fingerprints
        fingerprints.append(fingerprint)
    # Total Tanimoto Distance
    tanimoto = 0
    random_fingerprints = random.sample(fingerprints, num_random_molecules)
    # Calculate Tanimoto Distance between each pair of fingerprints
    for fpt1 in random_fingerprints:
        for fpt2 in random_fingerprints:
            if fpt1 != fpt2:
                # Calculate Tanimoto Similarity
                tan = TanimotoSimilarity(fpt1, fpt2)
                tanimoto += tan
    # Average Tanimoto Similarity
    avg_tanimoto = (1 / (num_random_molecules * (num_random_molecules - 1))) * tanimoto
    print("Average Tanimoto Similarity: {:0.4f}".format(avg_tanimoto))

    # Combine best molecules with those generated in this iteration (maintains the best molecules throughout for faster
    # convergence)
    filenames = ["genmols.txt", "finestmols.txt"]
    with open(os.path.join(results_folder, "allmols.txt"), "w") as outfile:
        for file in filenames:
            path = os.path.join(results_folder, file)
            if os.path.exists(path):
                with open(path) as infile:
                    for line in infile:
                        outfile.write(line)
    # Array of molecular properties for generated molecules
    molProps = np.empty((0, 5))
    fine_gen_molecules = open(os.path.join(results_folder, "fgenmols.txt"), "w")
    # Read in data file line by line
    for molecule in open(os.path.join(results_folder, "genmols.txt"), "r"):
        try:
            # Array of properties
            props = np.reshape(np.array([log_p(molecule), mol_wt(molecule), num_acc(molecule), num_don(molecule),
                                         rol_bon(molecule)]), (1, 5))
            # Append properties
            molProps = np.append(molProps, props, axis=0)
            # Write molecules to final, cleaned dataset
            fine_gen_molecules.write(molecule)
        except:
            # Occasionally RDKit bugs don't allow for analyzing the molecule; in these cases, do not include the
            # molecule in the final dataset
            continue
    fine_gen_molecules.close()
    # Array of number of molecules each molecule is dominated by
    dom = np.zeros((np.shape(molProps)[0]))
    # Analyze each molecule's properties as they compare to others
    for i in range(np.shape(molProps)[0]):
        for j in range(np.shape(molProps)[0]):
            # Compare each property between the molecules
            if all((molProps[j, k] <= molProps[i, k]) for k in range(np.shape(molProps)[1])) and (
                    np.array_equal(molProps[i, :], molProps[j, :]) is False):
                # if molecule j is better than or equal to molecule i in every property, but not equal to i, than j
                # dominates i
                dom[i] += 1
    # Fraction of molecules to be selected
    top = 0.5
    # Max number of molecules to be selected
    max_molecules = 10000
    # Select the best molecules based on fraction / max number
    if int(top * (np.shape(molProps)[0])) < max_molecules:
        # Select the best of the molecules
        finest_molecules = np.argpartition(dom, int(top * (np.shape(molProps)[0])))
    else:
        # Select the best of the molecules
        finest_molecules = np.argpartition(dom, max_molecules)
    finest_molecules = finest_molecules[:int(top * (np.shape(molProps)[0]))]
    # Save best molecules
    transfer_data = open(os.path.join(results_folder, "finestmols.txt"), "w")
    i = 0
    for line in open(os.path.join(results_folder, "fgenmols.txt"), "r"):
        if i in finest_molecules:
            # Append start token
            line = line.rjust(len(line) + 1, "G")
            # Write to file
            transfer_data.write(line)
        i += 1
    transfer_data.close()
    # Read in original training data file to get character list
    data = open(os.path.join(output_folder, "smiles.txt"), "r").read()
    # Create a list of the unique characters in the dataset
    chars = list(set(data))
    # Create array from characters in the dataset
    values = array(chars)
    print("Array of unique characters:")
    print(values)
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
    # Read in data file of best molecules
    data = open(os.path.join(results_folder, "finestmols.txt"), "r").read()
    # Create a list of the dataset, along with all the characters to ensure they are all represented
    datalist = list(data)
    datalist.extend(chars)
    # Create an array of the dataset
    data_array = array(datalist)
    # Fit one-hot encoding to data_array
    data_array = data_array.reshape(len(data_array), 1)
    # Fit encoder, remove all characters at the end leaving just the molecules
    ohe_fine_molecules = one_hot_encoder.fit_transform(data_array).astype(int)
    print("Size of one-hot encoded array of data: " + str(ohe_fine_molecules.shape))
    # Save ohe_fine_molecules as a (compressed) file
    np.savez_compressed(os.path.join(results_folder, "ohefinemols.npz"), ohe_fine_molecules)
    # Create integer transfer data
    int_fine_molecules = [np.where(r == 1)[0][0] for r in ohe_fine_molecules]
    # Save int_fine_molecules as a (compressed) file
    np.savez_compressed(os.path.join(results_folder, "intfinemols.npz"), int_fine_molecules)
