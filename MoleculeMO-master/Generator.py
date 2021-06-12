# Import packages
import numpy as np
import torch.nn.functional as F
import time
from Modules.config import *
from Modules.utils import time_elapsed


def generate(model, vocab, prime_string, char_to_gen, temperature):
    # SMILES character string
    mol = "G"
    # Get input tensor from prime string
    prediction = torch.from_numpy(vocab[np.where(vocab == str(prime_string))[0], :][:, 2:].astype(float)) \
        .view(1, 1, -1).cuda()
    hidden = model.init_states(num_layers, hidden_size)
    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    for i in range(char_to_gen):
        # Get input tensor
        inp = prediction[i, :, :].view(1, 1, -1).float()
        # Run on GPU if available
        if cuda:
            inp = inp.cuda()
        # Run model
        output, hidden = model(inp, hidden)
        # Apply softmax to convert output into probabilities
        output = F.softmax((output / temperature), dim=2)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1)
        top_i = torch.multinomial(output_dist, 1)[0]
        char = torch.from_numpy(vocab[top_i, 2:].astype(float)).view(1, 1, -1).cuda()
        # Update total prediction with the new character
        prediction = torch.cat((prediction, char), 0)
        # SMILES character predicted
        smile = vocab[top_i, 0]
        if smile == "\n":
            hidden = model.init_states(num_layers, hidden_size)
            if cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
        # Update character string
        mol = mol + str(smile)
        mol = mol.replace("G", "")
    return prediction, mol


def generate_molecules(model, vocab, temperature=1, char_to_gen=1000, runs=100, gen_filename='gen.txt'):
    # Run on GPU
    if cuda:
        model = model.cuda()
    model.eval()
    start_time = time.time()
    prime_string = "G"
    # GPU can't handle generating larger amounts of characters at once, so done in a loop
    for i in range(runs):
        # File to save generated molecules in
        new = open(os.path.join(results_folder, gen_filename), "a")
        prediction, mol = generate(model, vocab, prime_string, char_to_gen, temperature)
        # Add to file of generated molecules
        new.write(mol)
        hours, minutes, seconds = time_elapsed(start_time)
        print(f"SMILES run: {i:4d} saved. | Time elapsed: {hours:02d} h {minutes:02d} m {seconds:02d} s")
