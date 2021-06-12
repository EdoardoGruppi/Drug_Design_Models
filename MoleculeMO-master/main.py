# Import packages
from DataPreprocessing import pre_processing
import numpy as np
from Modules.config import *
from LSTM import Model, train, load_model
from Modules.utils import load_data, load_vocab
from Generator import generate_molecules
from DataPostprocessing import post_processing

# # TRAIN THE MODEL
# # Pre-process the data
# pre_processing(filename='mini.txt')
# # Load the data
# data, int_data = load_data("ohesmiles.npz", "intsmiles.npz")
# input_size = np.shape(data)[2]
# # Instantiate the model
# model = Model(input_size, hidden_size, num_layers, dropout)
# # Train the model
# train(model, data, int_data, epochs=1, model_name="network.pth", output_file='training.txt', new_session=True)

# # CONTINUE THE TRAINING
# # Load Model
# model = load_model(model, model_file='network.pth')
# # # Improve model
# # train(model, data, int_data, epochs=1, model_name='network.pth', output_file='training.txt', new_session=False)
# #
# # GENERATE MOLECULES
# # Load the data and vocab
# data, _ = load_data("ohesmiles.npz", single_input=True)
# vocab = load_vocab()
# # Initialize model and generation parameters
# input_size = np.shape(data)[2]
# # Generate molecules
# generate_molecules(model, vocab, temperature=1, char_to_gen=1000, runs=100, gen_filename='gen.txt')

# POST PROCESSING
post_processing(num_random_molecules=5)
#
# # FINE TUNING
# # Load the data
# data, int_data = load_data("ohefinemols.npz", "intfinemols.npz", folder=results_folder)
# # Initialize model parameters
# input_size = np.shape(data)[2]
# # Call model, set optimizer and loss function
# model = Model(input_size, hidden_size, num_layers, dropout)
# # Load Model
# model = load_model(model, model_file='network.pth')
# # Improve model
# train(model, data, int_data, epochs=1, model_name='network1.pth', output_file='training_tuned.txt', new_session=True)


