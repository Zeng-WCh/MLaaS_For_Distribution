import json
import torch
from torch.utils.data.dataset import Dataset


class JsonDataset(Dataset):
    def __init__(self, json_file):
        super(JsonDataset, self).__init__()
        self.filename = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]['label']
        data = self.data[idx]['data']
        data_ = list()
        data_.append(data['x'])
        data_.append(data['y'])
        # print(data_)

        # data looks like (x, y)
        # label will be either 0 or 1
        # data and label are both torch tensors
        return torch.FloatTensor(data_), torch.tensor(label)
