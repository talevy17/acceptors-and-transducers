import torch.nn as nn
import torch


class SimpleLstm(nn.Module):
    def __init__(self, embedding_dim, sequence_dim, hidden_dim, output_dim, batch_size):
        super(SimpleLstm, self).__init__()
        self.batch_size = batch_size
        self.sequence_dim = sequence_dim
        self.lstm = nn.LSTM(embedding_dim, sequence_dim)
        self.non_linear = nn.Sequential(nn.Linear(sequence_dim, hidden_dim), nn.Tanh())
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, sequence):
        pred = sequence.view(1, len(sequence), -1)
        pred = pred.permute(1, 0, 2)
        h_0 = torch.zeros(1, self.batch_size, self.sequence_dim, dtype=torch.float)
        c_0 = torch.zeros(1, self.batch_size, self.sequence_dim, dtype=torch.float)
        output, (final_hidden_state, final_cell_state) = self.lstm(pred, (h_0, c_0))
        pred = self.non_linear(final_hidden_state[-1])
        pred = self.linear(pred)
        return self.softmax(pred[0])
