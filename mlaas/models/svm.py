import torch
import torchvision

import torch.nn as nn

from tqdm import tqdm


def hinge_loss(output, y):
    output = output.squeeze()
    return torch.mean(torch.clamp(1 - y * output, min=0))


class SVM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def create_SVM(args, kwargs):
    if len(args) > 0:
        input_dim = args[0]
        output_dim = args[1]
    else:
        input_dim = kwargs['input_dim']
        output_dim = kwargs['output_dim']
    return SVM(input_dim, output_dim)


if __name__ == '__main__':
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    # Use MNIST dataset
    model = SVM(28 * 28, 10)

    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True)

    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        test, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in tqdm(range(20)):
        loss = 0

        for inputs, labels in trainloader:
            inputs = inputs.view(-1, 28 * 28)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch {i} Loss: {loss.item()}')
