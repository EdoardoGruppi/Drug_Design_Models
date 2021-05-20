# Import packages
from train import engine, engine_cond
from test import test_mol_rnn, test_cond_mol_rnn

# Create and train the unconditional molRNN
params = {'checkpoint_dir': 'checkpoint/mol_rnn', 'is_full': False, 'num_folds': 5, 'fold_id': 0, 'batch_size': 10,
          'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8, 'embedding_size': 16, 'hidden_sizes': [32, 64],
          'skip_layer_size': 64, 'dense_layer_sizes': [64], 'policy_layers_size': 32, 'rnn_layers': 2,
          'activation': 'relu', 'gpu_ids': [0], 'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0,
          'iterations': 180, 'summary_step': 20, 'file_name': 'datasets/ChEMBL.txt'}
engine(**params)
# Test the unconditional molRNN
test_mol_rnn(checkpoint_folder='checkpoint/mol_rnn', num_samples=1000, visualize_samples=12, img_name='image000',
             molecules_per_row=3, plot=True)

# Create and train the molRNN conditioned on SA and QED properties
params = {'cond_type': 'prop', 'checkpoint_dir': 'checkpoint/cp_mol_rnn', 'is_full': False, 'num_folds': 5,
          'fold_id': 0, 'batch_size': 10, 'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8,
          'embedding_size': 16, 'hidden_sizes': [32, 64], 'skip_layer_size': 64, 'dense_layer_sizes': [64],
          'policy_layers_size': 32, 'rnn_layers': 2, 'num_scaffolds': 0, 'activation': 'relu', 'gpu_ids': [0],
          'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0, 'iterations': 200, 'summary_step': 20,
          'file_name': 'datasets/ChEMBL_prop.txt'}
engine_cond(**params)
# Test the molRNN conditioned on SA and QED properties
c = [[0.84, 1.9],
     [0.27, 2.5],
     [0.84, 3.8],
     [0.27, 4.8]]
test_cond_mol_rnn(c, checkpoint_folder='checkpoint/cp_mol_rnn/', num_samples=1000, visualize_samples=12,
                  img_name='cp_image00', molecules_per_row=4, plot=True, legend='prop')

# Create and train the molRNN conditioned on measured activity towards molecules
params = {'cond_type': 'kinase', 'checkpoint_dir': 'checkpoint/ck_mol_rnn', 'is_full': False, 'num_folds': 5,
          'fold_id': 0, 'batch_size': 10, 'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8,
          'embedding_size': 16, 'hidden_sizes': [32, 64], 'skip_layer_size': 64, 'dense_layer_sizes': [64],
          'policy_layers_size': 32, 'rnn_layers': 2, 'num_scaffolds': 0, 'activation': 'relu', 'gpu_ids': [0],
          'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0, 'iterations': 200, 'summary_step': 20,
          'file_name': 'datasets/ChEMBL_k.txt'}
engine_cond(**params)
# Test the molRNN conditioned on measured activity towards molecules
c = [[1, 1],  # GSK-3b(+) & JNK3(+)
     [1, 0],  # GSK-3b(+) & JNK3(-)
     [0, 1]]  # GSK-3b(-) & JNK3(+)
test_cond_mol_rnn(c, checkpoint_folder='checkpoint/ck_mol_rnn/', num_samples=1000, visualize_samples=12,
                  img_name='ck_image00', molecules_per_row=4, plot=True)
