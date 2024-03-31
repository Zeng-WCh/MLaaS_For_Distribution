import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from mlaas.models.svm import SVM


def read_json(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    result = list()
    label = list()
    for part in data:
        result.append(np.array([part["data"]["x"], part["data"]["y"]]))
        label.append((part["label"]))
    return np.array(result), np.array(label)


def visualize(X, model, output_path):
    W = model.linear.weight.squeeze().detach().cpu().numpy()
    b = model.linear.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Spectral")
    plt.scatter(x=X[:, 0], y=X[:, 1], s=10, c="black",)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default="SVM.pt")
    parser.add_argument('--data_path', type=str, default="data.json")
    parser.add_argument('--output', type=str, default="SVM_Result.png")

    args = parser.parse_args()

    X, Y = read_json(args.data_path)
    model = SVM(2, 1)
    model.load_state_dict(torch.load(args.model_path))
    visualize(X, model, args.output)
