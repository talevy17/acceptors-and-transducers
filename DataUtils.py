import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataParser:
    def __init__(self, mode, F2I={}):
        self.sequences, self.labels = self.read_data(mode)
        self.F2I = F2I if F2I \
            else {f: i for i, f in enumerate(list(sorted(set([char for seq in self.sequences for char in seq]))))}
        self.I2F = {i: f for f, i in self.F2I.items()}
        self.mode = mode
        self.convert_to_indexes()

    @staticmethod
    def read_data(mode):
        with open('./Data/{0}'.format(mode), 'r') as file:
            data = file.readlines()
        sequences = []
        labels = []
        for row in data:
            parsed = row.split('\n')
            parsed = parsed[0].split('\t')
            sequences.append([char for char in parsed[0]])
            if not mode == 'test':
                labels.append(int(parsed[1]))
        return sequences, labels

    def convert_to_indexes(self):
        for sequence in self.sequences:
            for index, char in enumerate(sequence):
                sequence[index] = self.F2I[char]

    @staticmethod
    def tensor_conversion(data):
        ret = torch.from_numpy((np.asarray(data)))
        ret = ret.type(torch.long)
        return ret

    def data_loader(self, batch_size=1, shuffle=True):
        sequences = self.tensor_conversion(self.sequences)
        labels = self.tensor_conversion(self.labels)
        return DataLoader(TensorDataset(sequences, labels), batch_size, shuffle=shuffle) if not self.mode == "test" \
            else DataLoader(TensorDataset(sequences), batch_size, shuffle=shuffle)

    def get_F2I(self):
        return self.F2I

    def get_I2F(self):
        return self.I2F