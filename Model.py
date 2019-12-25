import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, sequence_dim, hidden_dim, output_dim, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sequence_dim = sequence_dim
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.lstm = nn.LSTM(embedding_dim, sequence_dim)
        self.non_linear = nn.Sequential(nn.Linear(sequence_dim, hidden_dim), nn.Dropout(), nn.ReLU())
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, batch_size=None):
        pred = self.embed(sequence)
        pred = pred.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        h_0 = torch.zeros(1, self.batch_size, self.sequence_dim)
        c_0 = torch.zeros(1, self.batch_size, self.sequence_dim)
        output, (final_hidden_state, final_cell_state) = self.lstm(pred, (h_0, c_0))
        pred = self.non_linear(final_hidden_state[-1])
        pred = self.linear(pred)
        return pred

