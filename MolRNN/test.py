from mx_mg import builders
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import random
import matplotlib.pyplot as plt
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED
import os


def visualize_molecules(samples, img_name, molecules_per_row=3, plot=True, legend=None):
    """
    Plot and/or save an image of the molecules achieved.

    :param samples: results obtained from the trained model.
    :param img_name: name of the image to save.
    :param molecules_per_row: number of molecules to put in every row of the image.
    :param plot: boolean, if True the image is also shown.
    :param legend: image legends.
    :return:
    """
    image = Draw.MolsToGridImage(samples, molsPerRow=molecules_per_row, legends=legend)
    image.save(os.path.join('images', img_name + '.png'))
    if plot:
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def visualize_molecules_cond(samples, visualize_samples, img_name, molecules_per_row=4, plot=True, legend=None):
    """
    Plot and/or save an image of the molecules achieved with the conditioned MolRNN.

    :param samples: results obtained from the trained model.
    :param visualize_samples: number of samples to visualize.
    :param img_name: name of the image to save.
    :param molecules_per_row: number of molecules to put in every row of the image.
    :param plot: boolean, if True the image is also shown.
    :param legend: image legends. If equal to 'prop' the QED and SA scores are plotted as well.
    :return:
    """
    # Shuffle the results and remove the ones that are equal to None
    random.shuffle(samples)
    samples = [mol for mol in samples if mol is not None]
    if legend is 'prop':
        # Crete the legends of the images with the QED and SA scores
        legend = [f'QED={QED.qed(m):.2f},\n SAscore={sascorer.calculateScore(m):.2f}' for m in
                  samples[:visualize_samples]]
    # Plot the molecules obtained
    visualize_molecules(samples[:visualize_samples], img_name, molecules_per_row, plot, legend)


def test_mol_rnn(checkpoint_folder='checkpoint/mol_rnn/', num_samples=1000, visualize_samples=12, img_name='image000',
                 molecules_per_row=3, plot=True):
    """
    Test the MolRNN unconditional model.

    :param checkpoint_folder: location where the training results and model are stored.
    :param num_samples: number of molecules to generate.
    :param visualize_samples: number of molecules to visualize.
    :param img_name: name of the png file to create.
    :param molecules_per_row: number of molecules to put in every row of the image.
    :param plot: boolean, if True the image is also shown.
    :return:
    """
    # Load the pre-trained model
    mol_rnn = builders.Vanilla_RNN_Builder(checkpoint_folder, gpu_id=0)
    # Generate new molecules
    samples_mol_rnn = [mol for mol in mol_rnn.sample(num_samples) if mol is not None]
    # Shuffle the list of molecules generated
    random.shuffle(samples_mol_rnn)
    # Visualize some of the new molecules
    visualize_molecules(samples_mol_rnn[:visualize_samples], img_name, molecules_per_row, plot=plot)


def test_cond_mol_rnn(conditional_codes, checkpoint_folder='checkpoint/cp_mol_rnn/', num_samples=1000,
                      visualize_samples=12, img_name='image00', molecules_per_row=4, plot=True, legend=None):
    """
    Test the conditioned MolRNN model.

    :param conditional_codes: list representing the conditional codes.
    :param checkpoint_folder: location where the training results and model are stored.
    :param num_samples: number of molecules to generate.
    :param visualize_samples: number of molecules to visualize.
    :param img_name: name of the png file to create.
    :param molecules_per_row: number of molecules to put in every row of the image.
    :param plot: boolean, if True the image is also shown.
    :param legend: image legends. If equal to 'prop' the QED and SA scores are plotted as well.
    :return:
    """
    conditional_codes = np.array(conditional_codes, dtype=np.float32)
    sectors = conditional_codes.shape[0]
    # Load the conditional MolRNN model
    model = builders.CVanilla_RNN_Builder(checkpoint_folder, gpu_id=0)
    # Sample the results
    prop_outputs = []
    for num in range(sectors):
        # For each sector of values save the results as a list of molecules
        samples_num = [mol for mol in model.sample(num_samples, c=conditional_codes[num, :], output_type='mol') if
                       mol is not None]
        # Convert the molecules into smiles strings
        smiles_list = [Chem.MolToSmiles(mol) for mol in samples_num]
        # Remove all the duplicates
        smiles_list = list(set(smiles_list))
        # Convert back the strings into mol objects
        samples_num = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        # Shuffle the list of molecules and append it to the dedicated list
        random.shuffle(samples_num)
        prop_outputs.append(samples_num)
    # For every list of results related to each sector display the achieved molecules
    for num in range(sectors):
        visualize_molecules_cond(prop_outputs[num], visualize_samples, img_name=img_name + str(num),
                                 molecules_per_row=molecules_per_row, plot=plot, legend=legend)
