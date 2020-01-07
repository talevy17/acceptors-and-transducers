from BiLSTM import BiLSTM as Model
from ModelIterator import iterate_model
from DataUtils import DataReader, NONE, UNKNOWN
import sys


def a(train_file=None, model_file=None, dev=True):
    data_type = "pos"
    train_data = DataReader(data_type=data_type, mode="train", train_file=train_file)
    F2I = train_data.get_f2i()
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    I2F = train_data.get_i2f()
    vocab_size = len(F2I)
    embedding_dim = 300
    hidden_dim = 1000
    batch_size = 20
    output_dim = len(L2I)
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr='a', I2L=I2L, I2F=I2F)
    if dev:
        dev_data = DataReader(data_type=data_type, mode="dev", F2I=F2I, L2I=L2I)
        model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="a-{0}".format(data_type), batch_size=batch_size)
    else:
        model = iterate_model(model, train_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="a-{0}".format(data_type), batch_size=batch_size)
    if model_file is not None:
        model.save(model_file)


def b_d(choice, train_file=None, model_file=None, dev=True):
    data_type = "ner"
    train_data = DataReader(data_type=data_type, mode="train", train_file=train_file)
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
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr=choice,
                  letter_dict=letter_dict, I2F=I2F, word_len=max_len, char_dim=char_dim, I2L=I2L)
    if dev:
        dev_data = DataReader(data_type=data_type, mode="dev", F2I=F2I, L2I=L2I)
        model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="{0}-{1}".format(choice, data_type), batch_size=batch_size)
    else:
        model = iterate_model(model, train_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="{0}-{1}".format(choice, data_type), batch_size=batch_size)
    if model_file is not None:
        model.save(model_file)


def create_dictionaries(prefix_size, suffix_size, sentences, i2f):
    words = [word for sentence in sentences for word in sentence]
    prefixes = [i2f[word][: prefix_size] for word in words]
    suffixes = [i2f[word][-suffix_size:] for word in words]
    PRE2I = {pre: i for i, pre in enumerate(sorted(set(prefixes)))}
    PRE2I[UNKNOWN] = len(PRE2I)
    SUF2I = {suf: i for i, suf in enumerate(sorted(set(suffixes)))}
    SUF2I[UNKNOWN] = len(SUF2I)
    return PRE2I, SUF2I


def c(train_file=None, model_file=None, dev=True):
    data_type = "pos"
    train_data = DataReader(data_type=data_type, mode="train", train_file=train_file)
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
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I, repr='c',
                  PREF2I=PRE2I, SUFF2I=SUF2I, I2F=I2F, I2L=I2L)
    if dev:
        dev_data = DataReader(data_type=data_type, mode="dev", F2I=F2I, L2I=L2I)
        model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="c-{0}".format(data_type), batch_size=batch_size)
    else:
        model = iterate_model(model, train_data.data_loader(batch_size), I2L=I2L,
                              ignore_index=L2I[NONE], mode="c-{0}".format(data_type), batch_size=batch_size)
    if model_file is not None:
        model.save(model_file)


def handle_args():
    choice = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]
    if choice == 'a':
        a(train_file, model_file, dev=False)
    elif choice == 'c':
        c(train_file, model_file, dev=False)
    elif choice == 'b' or choice == 'd':
        b_d(choice, train_file, model_file, dev=False)


if __name__ == "__main__":
    c()
    # handle_args()



