import torch.nn as nn
import torch


class Model(nn.Module):

    def __init__(self, embedding_dim, vocab_size, sequence_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, sequence_dim)
        torch.manual_seed(3)
        self.embed = nn.Embedding(embedding_dim, vocab_size)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.non_linear = nn.Sequential(nn.Linear(sequence_dim, hidden_dim), nn.Tanh())
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence):
        data = self.embed(sequence).view(-1, len(sequence))
        data, _ = self.lstm(data.view(len(data), 1, -1))
        data = self.non_linear(data)
        data = self.linear(data)
        return self.softmax(data)