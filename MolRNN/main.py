# Import packages
from train import _engine, _engine_cond

# Create and train the unconditional molRNN
params = {'checkpoint_dir': 'checkpoint/mol_rnn', 'is_full': False, 'num_folds': 5, 'fold_id': 0, 'batch_size': 10,
          'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8, 'embedding_size': 16, 'hidden_sizes': [32, 64],
          'skip_layer_size': 64, 'dense_layer_sizes': [64], 'policy_layers_size': 32, 'rnn_layers': 2,
          'activation': 'relu', 'gpu_ids': [0], 'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0,
          'iterations': 140, 'summary_step': 20, 'file_name': 'datasets/ChEMBL.txt'}
_engine(**params)

# Create and train the molRNN conditioned on SA and QED properties
params = {'cond_type': 'prop', 'checkpoint_dir': 'checkpoint/cp_mol_rnn', 'is_full': False, 'num_folds': 5,
          'fold_id': 0, 'batch_size': 10, 'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8,
          'embedding_size': 16, 'hidden_sizes': [32, 64], 'skip_layer_size': 64, 'dense_layer_sizes': [64],
          'policy_layers_size': 32, 'rnn_layers': 2, 'num_scaffolds': 0, 'activation': 'relu', 'gpu_ids': [0],
          'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0, 'iterations': 120, 'summary_step': 20,
          'file_name': 'datasets/ChEMBL_prop.txt'}
_engine_cond(**params)

# Create and train the molRNN conditioned on measured activity towards molecules
params = {'cond_type': 'kinase', 'checkpoint_dir': 'checkpoint/ck_mol_rnn', 'is_full': False, 'num_folds': 5,
          'fold_id': 0, 'batch_size': 10, 'batch_size_test': 10, 'num_workers': 0, 'k': 5, 'p': 0.8,
          'embedding_size': 16, 'hidden_sizes': [32, 64], 'skip_layer_size': 64, 'dense_layer_sizes': [64],
          'policy_layers_size': 32, 'rnn_layers': 2, 'num_scaffolds': 0, 'activation': 'relu', 'gpu_ids': [0],
          'lr': 0.001, 'decay': 0.01, 'decay_step': 100, 'clip_grad': 3.0, 'iterations': 100, 'summary_step': 20,
          'file_name': 'datasets/ChEMBL_k.txt'}
_engine_cond(**params)
