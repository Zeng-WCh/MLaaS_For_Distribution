import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import json
import torch

# Generate toy data that has two distinct classes and a huge gap between them
X, Y = make_blobs(n_samples=5000, centers=2, random_state=0,
                  cluster_std=0.4)  # X - features, Y - labels
Y[Y == 0] = -1
# X = (X - X.mean())/X.std()
# Plot the toy data
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
plt.savefig('data.png')

perm = torch.randperm(len(X))
X = torch.FloatTensor(X)
x = X[perm[0:2]]
print(torch.autograd.Variable(x))

output = list()

# Save the toy data to a file
for i in range(len(X)):
    point = dict()
    d = X[i].tolist()
    point["data"] = {"x": d[0], "y": d[1]}
    point["label"] = int(Y[i])
    output.append(point)


with open("data.json", "w") as f:
    json.dump(output, f, indent=4)
