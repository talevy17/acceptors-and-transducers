import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, NONE, repr, word_dim=None):
        super(BiLSTM, self).__init__()
        self.NONE = NONE
        self.batch_size = batch_size
        self.F2I = F2I
        self.repr = repr
        self.hidden = hidden_dim
        dimension = word_dim if word_dim else embedding_dim
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, dimension)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        if repr == 'b':
            self.lstm_embedding = nn.LSTM(word_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.output_dim = output_dim
        self.softmax = nn.LogSoftmax(dim=2)

    def calc_lengths(self, batch):
        lengths = []
        for sentence in batch:
            index = 0
            while not index == len(sentence) and not int(sentence[index]) == int(self.F2I[self.NONE]):
                index += 1
            if not index == 0:
                lengths.append(index)
            else:
                lengths.append(1)
        return lengths

    def forward(self, sentence):
        embedded = self.embed(sentence)
        if self.repr == 'b':
            embedded = self.lstm_embedding(embedded)
        # seq_lengths = self.calc_lengths(sentence)
        feed = embedded.permute(1, 0, 2)
        # packed = pack_padded_sequence(feed, seq_lengths, enforce_sorted=False)
        output, _ = self.lstm(feed)
        # output, _ = pad_packed_sequence(output, padding_value=0, total_length=len(sentence[0]))
        pred = self.linear(output)
        return self.softmax(pred)
