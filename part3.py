from BiLSTM import BiLSTM as Model
from ModelIterator import iterate_model
from DataUtils import DataReader


def check_model():
    train_data = DataReader(data_type="pos", mode="train")
    F2I = train_data.get_f2i()
    L2I = train_data.get_l2i()
    vocab_size = len(F2I)
    embedding_dim = 300
    hidden_1 = 1000
    hidden_2 = 300
    batch_size = 1000
    output_dim = len(L2I)
    dev_data = DataReader(data_type="pos", mode="train", F2I=F2I, L2I=L2I)
    model = Model(embedding_dim, vocab_size, hidden_1, hidden_2, output_dim, batch_size)
    model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size))


if __name__ == "__main__":
    check_model()