# Import packages
import torch
import os

# Set device
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False

print("Device =", device)
gpus = [0]

# Input -------------------------------------
output_folder = 'Vocab'
model_folder = 'Model'
results_folder = 'Results'
# -------------------------------------------


# Network parameters ------------------------
hidden_size = 1024
num_layers = 3
dropout = .2
learning_rate = 0.001
# Maximum size of char to
seq_length = 75
batch_size = 64
# List of losses
losses = [0]
# ---------------------------------------------
