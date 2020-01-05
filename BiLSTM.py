import torch.nn as nn
import torch
import numpy as np


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, NONE, repr,
                 I2F={}, PREF2I={}, SUFF2I={}, letter_dict={}, word_len=None, char_dim=None):
        super(BiLSTM, self).__init__()
        self.NONE = NONE
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.F2I = F2I
        self.repr = repr
        self.hidden = hidden_dim
        if not repr == 'b':
            torch.manual_seed(3)
            self.embed = nn.Embedding(vocab_size, embedding_dim)
            nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        if repr == 'b' or repr == 'd':
            torch.manual_seed(3)
            self.char_embed = nn.Embedding(vocab_size, char_dim)
            nn.init.uniform_(self.char_embed.weight, -1.0, 1.0)
            self.lstm_embedding = nn.LSTM(char_dim, embedding_dim, batch_first=True)
            self.I2F = I2F
            self.letter_dict = letter_dict
            self.word_len = word_len
        elif repr == 'c':
            self.PRE2I = PREF2I
            self.SUF2I = SUFF2I
            self.I2F = I2F
            self.embed_prefix = nn.Embedding(len(PREF2I), embedding_dim)
            nn.init.uniform_(self.embed_prefix.weight, -1.0, 1.0)
            self.embed_suffix = nn.Embedding(len(SUFF2I), embedding_dim)
            nn.init.uniform_(self.embed_suffix.weight, -1.0, 1.0)
        if repr == 'd':
            self.concat = nn.Linear(2 * embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.output_dim = output_dim
        self.softmax = nn.LogSoftmax(dim=2)

    @staticmethod
    def prepare_list(str_list, max_length, mapper, padding):
        idx_list = []
        for s in str_list:
            if str_list == padding:
                break
            if s in mapper:
                idx_list.append(mapper[s])
            else:
                idx_list.append(mapper[padding])
        while len(idx_list) < max_length:
            idx_list.append(mapper[padding])
        return idx_list

    def make_letter_input(self, sentences):
        # input shape is (batch_size, num_sequences)
        word_input = sentences.view(-1)
        # input shape is (batch_size * num_sequences)
        letter_input = torch.LongTensor(len(word_input), self.word_len)
        for i, idx in enumerate(word_input):
            word = self.I2F[int(idx)]
            letter_input[i] = torch.LongTensor(self.prepare_list(word, self.word_len, self.letter_dict, self.NONE))
        return letter_input

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
        if self.repr == 'a' or self.repr == 'd':
            embedded = self.embed(sentences)
        if self.repr == 'b' or self.repr == 'd':
            letter_input = self.make_letter_input(sentences)
            embedded = self.char_embed(letter_input)
            out, (f_h, f_c) = self.lstm_embedding(embedded)
            if self.repr == 'd':
                b_embedded = f_h.view(len(sentences), len(sentences[0]), self.embedding_dim)
                embedded = self.concat(torch.cat((embedded, b_embedded), 2))
            else:
                embedded = f_h.view(len(sentences), len(sentences[0]), self.embedding_dim)
        if self.repr == 'c':
            embedded = self.embed(sentences)
            prefix, suffix = self.make_prefix_suffix_input(sentences)
            prefix_embed = self.embed_prefix(prefix)
            suffix_embed = self.embed_suffix(suffix)
            embedded = embedded + prefix_embed + suffix_embed
        feed = embedded.permute(1, 0, 2)
        output, _ = self.lstm(feed)
        pred = self.linear(output)
        return self.softmax(pred)

