from BiLSTM import BiLSTM as Model
from ModelIterator import iterate_model
from DataUtils import DataReader, NONE


def a():
    train_data = DataReader(data_type="pos", mode="train")
    F2I = train_data.get_f2i()
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    vocab_size = len(F2I)
    embedding_dim = 300
    hidden_dim = 1000
    batch_size = 20
    output_dim = len(L2I)
    dev_data = DataReader(data_type="pos", mode="dev", F2I=F2I, L2I=L2I)
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr='a')
    model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                          ignore_index=L2I[NONE])


def b():
    train_data = DataReader(data_type="pos", mode="train")
    F2I = train_data.get_f2i()
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    I2F = train_data.get_i2f()

    letter_dict = train_data.get_char_dict()
    vocab_size = len(letter_dict)
    embedding_dim = 150
    hidden_dim = 400
    batch_size = 50
    char_dim = 30
    max_len = 8
    output_dim = len(L2I)
    dev_data = DataReader(data_type="pos", mode="dev", F2I=F2I, L2I=L2I)
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr='b',
                  letter_dict=letter_dict, I2F=I2F, word_len=max_len, char_dim=char_dim)
    model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                          ignore_index=L2I[NONE])


def create_dictionaries(prefix_size, suffix_size, sentences, i2f):
    words = [word for sentence in sentences for word in sentence]
    prefixes = [i2f[word][: prefix_size] for word in words]
    suffixes = [i2f[word][-suffix_size:] for word in words]
    PRE2I = {pre: i for i, pre in enumerate(sorted(set(prefixes)))}
    SUF2I = {suf: i for i, suf in enumerate(sorted(set(suffixes)))}
    return PRE2I, SUF2I


def c():
    train_data = DataReader(data_type="pos", mode="train")
    F2I = train_data.get_f2i()
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    I2F = train_data.get_i2f()
    PRE2I, SUF2I = create_dictionaries(3, 3, train_data.get_sentences(), I2F)
    vocab_size = len(F2I)
    embedding_dim = 300
    hidden_dim = 1000
    batch_size = 20
    output_dim = len(L2I)
    dev_data = DataReader(data_type="pos", mode="dev", F2I=F2I, L2I=L2I)
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr='c',
                  PREF2I=PRE2I, SUFF2I=SUF2I, I2F=I2F)
    model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                          ignore_index=L2I[NONE])


if __name__ == "__main__":
    b()
