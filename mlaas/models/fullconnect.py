import torch
import torch.nn as nn


class FullConnect(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, output_dim):
        super(FullConnect, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, output_dim)

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, self.input_dim)
        # Add activation function
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_FullConnect(args, kwargs):
    if len(args) > 0:
        input_dim = args[0]
        n_hidden_1 = args[1]
        n_hidden_2 = args[2]
        output_dim = args[3]
    else:
        input_dim = kwargs['input_dim']
        n_hidden_1 = kwargs['n_hidden_1']
        n_hidden_2 = kwargs['n_hidden_2']
        output_dim = kwargs['output_dim']
    return FullConnect(input_dim, n_hidden_1, n_hidden_2, output_dim)
