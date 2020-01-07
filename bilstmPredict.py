import sys
from DataUtils import DataReader
from ModelIterator import predict
from BiLSTM import BiLSTM as Model


def main():
    choice = sys.argv[1]
    model_file = sys.argv[2]
    input_file = sys.argv[3]
    model = Model.load(model_file)
    i2f = model.get_i2f()
    i2l = model.get_i2l()
    f2i = model.get_f2i()
    l2i = model.get_l2i()
    test_data = DataReader(mode="test", F2I=f2i, L2I=l2i, train_file=input_file)
    predict(model, test_data.data_loader(batch_size=1, shuffle=False), I2L=i2l, I2F=i2f, data_type=choice)


if __name__ == "__main__":
    main()
