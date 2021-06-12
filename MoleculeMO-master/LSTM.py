# Import packages
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
from Modules.config import *
from Modules.utils import time_elapsed, input_tensors, input_tensors_new
import re
from math import floor
from Modules.utils import load_data

# Define model
class Model(nn.Module):

    # Define model parameters
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Model, self).__init__()
        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Model layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, self.input_size)

    # Define initial hidden and cell states
    def init_states(self, num_layers, hidden_size):
        hidden = [Variable(torch.zeros(num_layers, 1, hidden_size)),
                  Variable(torch.zeros(num_layers, 1, hidden_size))]
        # Initialize forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                nn.init.constant_(bias.data[start:end], 1.0)
        return hidden

    # Define forward propagation
    def forward(self, inp, hidden):
        # LSTM
        output, hidden = self.lstm(inp, hidden)
        # Linear Layer
        output = self.linear(output)
        return output, hidden


# Define training
def train_epoch(model, data_list, int_data_list, optimizer, criterion, model_name='network.pth', output_file='training.txt',
                count=0, t_final=0):

    for data_filename, int_filename in zip(data_list, int_data_list):
        data, int_data = load_data(data_filename, int_filename)
        file = open(os.path.join(model_folder, output_file), "a")
        # Set start time
        start_time = time.time() - t_final
        print(f'Each epoch has {int(floor(np.shape(data)[0] / (batch_size * seq_length)))} batches, aka iterations.')
        # Iterate set of seq_length characters
        for i in range(int(floor(np.shape(data)[0] / (batch_size * seq_length)))):
            # Initialize hidden and cell states
            hidden = model.init_states(num_layers, hidden_size)
            # Run on GPU if available
            if cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            # Set initial gradients
            model.zero_grad()
            # Set initial loss
            loss = 0
            # Get input and target
            input_data, target_data = input_tensors_new(i, batch_size, seq_length, data, int_data)
            input_data = input_data.float()
            target_data = target_data.long()
            # Run on GPU if available
            if cuda:
                input_data = input_data.cuda()
                target_data = target_data.cuda()
            # Run model, calculate loss
            output, hidden = model(input_data, hidden)
            loss += criterion(output.squeeze(), target_data.squeeze())
            # Back propagate loss
            loss.backward()
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            # Optimize
            optimizer.step()
            # Update list of losses
            losses.append(loss.data.item())
            count += 1
            # Intermediary saves
            if i % 10 == 0:
                torch.save(model.state_dict(), os.path.join(model_folder, model_name))
                # Print training info
                hours, minutes, seconds = time_elapsed(start_time)
                output_string = f"Loss: {loss.data.item() / seq_length:0.6f} | Delta: {losses[-1] - losses[-2]:+0.4f} | " + \
                                f"Iteration: {count:06d} | Time elapsed: {hours:03d} h {minutes:02d} m {seconds:02d} s"
                print(output_string)
                file.write(output_string + '\n')
                file.flush()
    torch.save(model.state_dict(), os.path.join(model_folder, model_name))
    # Print training info
    hours, minutes, seconds = time_elapsed(start_time)
    output_string = f"Loss: {loss.data.item() / seq_length:0.6f} | Delta: {losses[-1] - losses[-2]:+0.4f} | " + \
                    f"Iteration: {count:06d} | Time elapsed: {hours:03d} h {minutes:02d} m {seconds:02d} s"
    print(output_string)
    file.write(output_string + '\n')
    file.flush()
    file.close()
    return count


def train(model, data, int_data, epochs, model_name, output_file='training.txt', new_session=False):
    if not new_session:
        with open(os.path.join(model_folder, output_file)) as f:
            last_record = f.readlines()[-1]
        count = int(re.findall('Iteration:(.*?)\|', last_record)[0].strip())
        t_final = [int(item) for item in re.findall('Time elapsed:(.*?)h(.*?)m(.*?)s', last_record)[0]]
        t_final = t_final[0] * 3600 + t_final[1] * 60 + t_final[2]
    else:
        file = open(os.path.join(model_folder, output_file), "w")
        file.close()
        count, t_final = 0, 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Run on GPU if available
    if cuda:
        model.cuda()
        criterion.cuda()
    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters in network: " + str(total_params))
    for i in range(epochs):
        # Train
        count = train_epoch(model, data, int_data, optimizer, criterion, model_name=model_name, output_file=output_file,
                    count=count, t_final=t_final)


def load_model(model, model_file="network.pth"):
    model.load_state_dict(torch.load(os.path.join(model_folder, model_file)))
    return model
