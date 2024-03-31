import torch

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from argparse import ArgumentParser
from mlaas.models.fullconnect import FullConnect

import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='full.pt')
    parser.add_argument('--output', type=str, default='full_result.png')
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()

    model = FullConnect(28 * 28, 300, 100, 10)

    test_loader = DataLoader(
        datasets.MNIST(args.data, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=1
    )

    model.load_state_dict(torch.load(args.mode_path))
    model.eval()

    # Get Accuracy
    correct = 0
    result = list()
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            pre = model(images)
            _, predicted = torch.max(pre.data, 1)
            total += labels.size(0)
            # print(images.squeeze(0).shape, labels.shape, predicted.shape)
            # exit()
            if predicted == labels:
                correct += 1
            result.append(
                (images.squeeze(0), labels.item(), predicted.item()))

            # correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total} %')

    # Plot 10 image and its prediction
    ax, fig = plt.subplots(2, 5, figsize=(20, 10))

    for i in range(10):
        idx = random.randint(0, len(result) - 1)
        img = result[idx][0].squeeze().numpy()
        label = result[idx][1]
        predicted = result[idx][2]

        fig[i // 5, i % 5].imshow(img, cmap='gray')
        fig[i // 5, i % 5].set_title(f'Label: {label}, Predict: {predicted}')

    plt.savefig(args.output)
    plt.show()
