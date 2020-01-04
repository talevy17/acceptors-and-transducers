import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, NONE, repr, char_dim=None):
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.output_dim = output_dim
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, sentences):
        if self.repr == 'b':
            embedded = []
            for sentence in sentences:
                ret = []
                for word in sentence:
                    out, _ = self.lstm_embedding(self.embed(word).view(len(word), 1, self.char_dim))
                    ret.append(out[-1])
                embedded.append(torch.cat(ret))
            embedded = torch.stack(embedded)
            # embedded = self.lstm_embedding(embedded)
        else:
            embedded = self.embed(sentences)
        # seq_lengths = self.calc_lengths(sentence)
        # packed = pack_padded_sequence(feed, seq_lengths, enforce_sorted=False)
        feed = embedded.permute(1, 0, 2)
        output, _ = self.lstm(feed)
        # output, _ = pad_packed_sequence(output, padding_value=0, total_length=len(sentence[0]))
        pred = self.linear(output)
        return self.softmax(pred)

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
