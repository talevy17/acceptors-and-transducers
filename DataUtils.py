import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataParser:
    def __init__(self, mode, F2I={}):
        self.sequences, self.labels = self.read_data(mode)
        self.F2I = F2I if F2I else self.create_dict(self.sequences)
        self.I2F = {i: f for f, i in self.F2I.items()}
        self.mode = mode
        self.sequence_dim = 0
        self.indexer_and_padding()

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

    def indexer_and_padding(self):
        for sequence in self.sequences:
            for index, char in enumerate(sequence):
                sequence[index] = self.F2I[char]
            if self.sequence_dim < len(sequence):
                self.sequence_dim = len(sequence)
        for index in range(len(self.sequences)):
            while len(self.sequences[index]) < self.sequence_dim:
                self.sequences[index].append(self.F2I['0'])

    @staticmethod
    def tensor_conversion(data, type):
        ret = torch.from_numpy((np.asarray(data)))
        ret = ret.type(type)
        return ret

    def data_loader(self, batch_size=1, shuffle=True):
        sequences = self.tensor_conversion(self.sequences, torch.long)
        labels = self.tensor_conversion(self.labels, torch.float)
        return DataLoader(TensorDataset(sequences, labels), batch_size, shuffle=shuffle) if not self.mode == "test" \
            else DataLoader(TensorDataset(sequences), batch_size, shuffle=shuffle)

    @staticmethod
    def create_dict(sequences):
        f2i = {f: i for i, f in enumerate(list(sorted(set([char for seq in sequences for char in seq]))))}
        f2i['0'] = len(f2i)
        return f2i

    def get_F2I(self):
        return self.F2I

    def get_sequence_dim(self):
        return self.sequence_dim

    def get_I2F(self):
        return self.I2F