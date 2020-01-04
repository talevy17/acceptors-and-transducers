import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, NONE, repr, char_dim=None,
                 I2F={}, PREF2I={}, SUFF2I={}):
        super(BiLSTM, self).__init__()
        self.NONE = NONE
        self.batch_size = batch_size
        self.F2I = F2I
        self.repr = repr
        self.hidden = hidden_dim
        self.char_dim = char_dim
        dimension = char_dim if char_dim else embedding_dim
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, dimension)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        if repr == 'b':
            self.lstm_embedding = nn.LSTM(char_dim, embedding_dim, batch_first=True)
        elif repr == 'c':
            self.PRE2I = PREF2I
            self.SUF2I = SUFF2I
            self.I2F = I2F
            self.embed_prefix = nn.Embedding(len(PREF2I), embedding_dim)
            nn.init.uniform_(self.embed_prefix.weight, -1.0, 1.0)
            self.embed_suffix = nn.Embedding(len(SUFF2I), embedding_dim)
            nn.init.uniform_(self.embed_suffix.weight, -1.0, 1.0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.output_dim = output_dim
        self.softmax = nn.LogSoftmax(dim=2)

    def one_hot(self, index):
        ret = np.zeros(len(self.F2I) + 1)
        ret[index] = int(1)
        return torch.from_numpy(ret).reshape(1, -1).type(torch.float)

    def encode_word(self, word):
        return torch.cat([self.one_hot(index) for index in word]).unsqueeze(0).type(torch.float)

    def get_prefix_index_by_word_index(self, index):
        return self.PRE2I[self.prefixes[index]]

    def get_suffix_index_by_word_index(self, index):
        return self.SUF2I[self.suffixes[index]]

    def make_prefix_suffix_input(self, sentences):
        # input shape is (batch_size, num_sequences)
        prefix_input = torch.LongTensor(len(sentences), len(sentences[0]))
        suffix_input = torch.LongTensor(len(sentences), len(sentences[0]))
        for i in range(len(sentences)):
            for j in range(len(sentences[0])):
                word = self.I2F[int(sentences[i][j])]
                prefix = word[:3]
                suffix = word[-3:]
                if prefix in self.PRE2I:
                    prefix_input[i][j] = self.PRE2I[prefix]
                else:
                    prefix_input[i][j] = self.PRE2I[self.NONE[:3]]
                if suffix in self.SUF2I:
                    suffix_input[i][j] = self.SUF2I[suffix]
                else:
                    suffix_input[i][j] = self.SUF2I[self.NONE[-3:]]
        return prefix_input, suffix_input

    def forward(self, sentences):
        if self.repr == 'b':
            embedded = []
            for sentence in sentences:
                ret = []
                for word in sentence:
                    out, _ = self.lstm_embedding(self.encode_word(word).view(len(word), 1, self.char_dim))
                    ret.append(out[-1])
                embedded.append(torch.cat(ret))
            embedded = torch.stack(embedded)
        else:
            embedded = self.embed(sentences)
            if self.repr == 'c':
                prefix, suffix = self.make_prefix_suffix_input(sentences)
                prefix_embed = self.embed_prefix(prefix)
                suffix_embed = self.embed_suffix(suffix)
                embedded = embedded + prefix_embed + suffix_embed
        feed = embedded.permute(1, 0, 2)
        output, _ = self.lstm(feed)
        pred = self.linear(output)
        return self.softmax(pred)

