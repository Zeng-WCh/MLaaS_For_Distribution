import torch
import torch.nn as nn
import torch.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def create_cnn(*args, **kwargs):
    return CNN()


if __name__ == '__main__':
    import torchvision
    from tqdm import tqdm
    import torch
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

    # Use MNIST dataset
    model = CNN()

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
