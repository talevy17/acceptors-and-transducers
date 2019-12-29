import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy


class ExperimentsParser:
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

    @staticmethod
    def tensor_conversion(data, type):
        ret = torch.from_numpy((np.asarray(data)))
        ret = ret.type(type)
        return ret

    def one_hot(self, index):
        ret = np.zeros(len(self.F2I))
        ret[index] = int(1)
        return torch.from_numpy(ret).reshape(1, -1).type(torch.float)

    def encode(self, sequence):
        return torch.cat([self.one_hot(index) for index in sequence]).type(torch.float)

    def encoder(self):
        return [self.encode(sequence) for sequence in self.sequences]

    def data_loader(self, batch_size=1, shuffle=True):
        sequences = self.tensor_conversion(self.sequences, torch.long)
        labels = self.tensor_conversion(self.labels, torch.long)
        return DataLoader(TensorDataset(sequences, labels), batch_size, shuffle=shuffle) if not self.mode == "test" \
            else DataLoader(TensorDataset(sequences), batch_size, shuffle=shuffle)

    @staticmethod
    def create_dict(sequences):
        return {f: i for i, f in enumerate(list(sorted(set([char for seq in sequences for char in seq]))))}

    def get_F2I(self):
        return self.F2I

    def get_sequence_dim(self):
        return self.sequence_dim

    def get_I2F(self):
        return self.I2F

    def get_labels(self):
        return self.labels


NONE = '*NONE*'


class DataReader:
    def __init__(self, data_type='pos', mode="train", F2I={}, L2I={}, to_lower=True):
        with open("./data/{0}/{1}".format(data_type, mode), 'r') as file:
            data = file.readlines()
        self.mode = mode
        self.sentence_len = 0
        self.sentences = []
        self.labels = []
        self.parse_sentences(data, data_type == 'pos', to_lower, mode)
        self.F2I = F2I if F2I else self.create_dict(self.sentences)
        self.L2I = L2I if L2I else self.create_dict(self.labels)
        self.convert_to_indexes()
        self.sentence_padding()

    def convert_to_indexes(self):
        for sentence, labels in zip(self.sentences, self.labels):
            for index, word in enumerate(sentence):
                if word in self.F2I:
                    sentence[index] = self.F2I[word]
                else:
                    sentence[index] = self.F2I[NONE]
            if not self.mode == "test":
                for index, label in enumerate(labels):
                    if label in self.L2I:
                        labels[index] = self.L2I[label]
                    else:
                        labels[index] = self.L2I[NONE]

    def parse_sentences(self, data, is_pos, to_lower, mode):
        # parse by spaces if post, if ner parse by tab.
        delimiter = ' ' if is_pos else '\t'
        current_sentence = []
        current_labels = []
        for row in data:
            row_spitted = row.split('\n')
            row_spitted = row_spitted[0].split(delimiter)
            word = row_spitted[0]
            if word != '':
                # convert all chars to lower case.
                if to_lower:
                    word = word.lower()
                if not mode == 'test':
                    label = row_spitted[1]
                    current_labels.append(label)
                current_sentence.append(word)
            else:
                if len(current_sentence) > self.sentence_len:
                    self.sentence_len = len(current_sentence)
                self.sentences.append(copy.deepcopy(current_sentence))
                self.labels.append(copy.deepcopy(current_labels))
                current_sentence.clear()
                current_labels.clear()

    def get_sentences(self):
        return self.sentences

    def get_labels(self):
        return self.labels

    def get_f2i(self):
        return self.F2I

    def get_l2i(self):
        return self.L2I

    @staticmethod
    def create_dict(data):
        data_dict = {f: i for i, f in enumerate(list(sorted(set([w for row in data for w in row]))))}
        data_dict[NONE] = len(data_dict)
        return data_dict

    def get_i2f(self):
        return {i: l for l, i in self.F2I.items()}

    def get_i2l(self):
        return {i: l for l, i in self.L2I.items()}

    @staticmethod
    def tensor_conversion(data):
        ret = torch.from_numpy((np.asarray(data)))
        ret = ret.type(torch.long)
        return ret

    def data_loader(self, batch_size=1, shuffle=True):
        windows = self.tensor_conversion(self.sentences)
        labels = self.tensor_conversion(self.labels)
        return DataLoader(TensorDataset(windows, labels), batch_size, shuffle=shuffle) if not self.mode == "test" \
            else DataLoader(TensorDataset(windows), batch_size, shuffle=shuffle)

    def sentence_padding(self):
        for sentence, label in zip(self.sentences, self.labels):
            while len(sentence) < self.sentence_len:
                sentence.append(self.F2I[NONE])
                label.append(self.L2I[NONE])
