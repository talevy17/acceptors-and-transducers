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
        self.indexer()

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

    def indexer(self):
        for sequence in self.sequences:
            for index, char in enumerate(sequence):
                sequence[index] = self.F2I[char]

    def one_hot(self, index):
        ret = np.zeros(len(self.F2I))
        ret[index] = int(1)
        return torch.from_numpy(ret).reshape(1, -1).type(torch.float)

    def encode(self, sequence):
        return torch.cat([self.one_hot(index) for index in sequence]).type(torch.float)

    def encoder(self):
        return [self.encode(sequence) for sequence in self.sequences]

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

CHAR_PAD = '**'


class DataReader:
    def __init__(self, data_type='pos', mode="train", F2I={}, L2I={}, to_lower=True, character_based=False):
        with open("./data/{0}/{1}".format(data_type, mode), 'r') as file:
            data = file.readlines()
        self.mode = mode
        self.character_based = character_based
        self.sentence_len = 0
        self.word_len = 0
        self.sentences = []
        self.labels = []
        self.parse_sentences(data, data_type == 'pos', to_lower, mode)
        if character_based:
            self.word_len = self.characterize()
            self.F2I = F2I if F2I else self.create_characterized_dict(self.sentences)
            self.L2I = L2I if L2I else self.create_dict(self.labels, CHAR_PAD)
            self.convert_chars_to_indexes()
            pad = []
            while len(pad) < self.word_len:
                pad.append(self.F2I[CHAR_PAD])
            self.sentence_padding(pad)

        else:
            self.F2I = F2I if F2I else self.create_dict(self.sentences, NONE)
            self.L2I = L2I if L2I else self.create_dict(self.labels, NONE)
            self.convert_to_indexes()
            self.sentence_padding(NONE)

    def characterize(self):
        max_len = 0
        for sentence in self.sentences:
            for index, word in enumerate(sentence):
                if len(word) > max_len:
                    max_len = len(word)
                sentence[index] = [char for char in word]
        for sentence in self.sentences:
            for word in sentence:
                while len(word) < max_len:
                    word.append(CHAR_PAD)
        return max_len

    @staticmethod
    def create_characterized_dict(data):
        data_dict = {f: i for i, f in enumerate(list(sorted(set([char for row in data for w in row for char in w]))))}
        data_dict[CHAR_PAD] = len(data_dict)
        return data_dict

    def convert_labels_batch(self, labels, skip):
        for index, label in enumerate(labels):
            if label in self.L2I:
                labels[index] = self.L2I[label]
            else:
                labels[index] = self.L2I[skip]

    def one_hot(self, index):
        ret = np.zeros(len(self.F2I) + 1)
        ret[index] = int(1)
        return torch.from_numpy(ret).reshape(1, -1).type(torch.float)

    def encode_word(self, word):
        return torch.cat([self.one_hot(index) for index in word]).unsqueeze(0).type(torch.float)

    def encode_sentence(self, sentence):
        return torch.cat([self.encode_word(word) for word in sentence]).unsqueeze(0).type(torch.float)

    def encoder(self):
        return [self.encode_sentence(sentence) for sentence in self.sentences]

    def convert_chars_to_indexes(self):
        for sentence, labels in zip(self.sentences, self.labels):
            for word in sentence:
                for index, char in enumerate(word):
                    if char in self.F2I:
                        word[index] = self.F2I[char]
                    else:
                        word[index] = self.F2I[CHAR_PAD]
            if not self.mode == "test":
                self.convert_labels_batch(labels, CHAR_PAD)

    def convert_to_indexes(self):
        for sentence, labels in zip(self.sentences, self.labels):
            for index, word in enumerate(sentence):
                if word in self.F2I:
                    sentence[index] = self.F2I[word]
                else:
                    sentence[index] = self.F2I[NONE]
            if not self.mode == "test":
                self.convert_labels_batch(labels, NONE)

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
    def create_dict(data, skip):
        data_dict = {f: i for i, f in enumerate(list(sorted(set([w for row in data for w in row]))))}
        data_dict[skip] = len(data_dict)
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

    def sentence_padding(self, skip):
        if self.character_based:
            word_index = skip
            label_index = self.L2I[CHAR_PAD]
        else:
            word_index = self.F2I[skip]
            label_index = self.L2I[skip]
        for sentence, label in zip(self.sentences, self.labels):
            while len(sentence) < self.sentence_len:
                sentence.append(word_index)
                label.append(label_index)

    def get_word_dim(self):
        return self.word_len
