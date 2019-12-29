import torch.nn as nn
import torch
import numpy as np


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim_1, hidden_dim_2, output_dim, batch_size):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden1 = hidden_dim_1
        self.hidden2 = hidden_dim_2
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim_1)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim_1)
        self.lstm2_f = nn.LSTM(hidden_dim_1, hidden_dim_2)
        self.lstm2_b = nn.LSTM(hidden_dim_1, hidden_dim_2)
        self.linear = nn.Linear(hidden_dim_2, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.softmax3 = nn.Softmax(dim=2)

    def init_hidden(self, dim):
        h_0 = torch.zeros(1, self.batch_size, dim)
        c_0 = torch.zeros(1, self.batch_size, dim)
        return h_0, c_0

    @staticmethod
    def flip_tensor(tensor):
        return torch.from_numpy(np.flip(tensor.detach().numpy(), 0).copy())

    def forward(self, sentence):
        embedded = self.embed(sentence)
        feed_f = embedded.permute(1, 0, 2)
        feed_b = self.flip_tensor(feed_f)
        hidden_f = self.init_hidden(self.hidden1)
        hidden_b = self.init_hidden(self.hidden1)
        output_f, (h_f, c_f) = self.lstm_f(feed_f, hidden_f)
        output_b, (h_b, c_b) = self.lstm_b(feed_b, hidden_b)
        # feed2_f = torch.cat((output_f, output_b))
        feed2_f = output_f + output_b
        feed2_b = self.flip_tensor(feed2_f)
        hidden2_f = self.init_hidden(self.hidden2)
        hidden2_b = self.init_hidden(self.hidden2)
        output2_f, (h2_f, c2_f) = self.lstm2_f(feed2_f, hidden2_f)
        output2_b, (h2_b, c2_b) = self.lstm2_b(feed2_b, hidden2_b)
        # pred = torch.cat((output2_f, output2_b))
        pred = output2_f + output2_b
        pred = self.linear(pred)
        return self.softmax3(pred)
