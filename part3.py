from BiLSTM import BiLSTM as Model
from ModelIterator import iterate_model
from DataUtils import DataReader, NONE


def check_model():
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
    model = Model(embedding_dim, vocab_size, hidden_dim, output_dim, batch_size, F2I)
    model = iterate_model(model, train_data.data_loader(batch_size), dev_data.data_loader(batch_size), I2L=I2L,
                          ignore_index=L2I[NONE])


if __name__ == "__main__":
    check_model()