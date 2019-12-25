import time
import torch
import torch.nn as nn
from DataUtils import DataParser
from Model import Model


def calc_batch_accuracy(predictions, labels):
    correct = wrong = 0
    for pred, label in zip(predictions, labels):
        if (pred > 0 and label == 1) or (pred >= 0 and label == 0):
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def train(model, loader, optimizer, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for index, batch in enumerate(loader):
        optimizer.zero_grad()
        predictions = model(batch[0].squeeze(1))
        loss = criterion(predictions[0], batch[1])
        acc = calc_batch_accuracy(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            predictions = model(batch[0].squeeze(1))
            loss = criterion(predictions[0], batch[1])
            acc = calc_batch_accuracy(predictions, batch[1])
            epoch_loss += loss.item()
            epoch_acc += acc
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def iterate_model(model, train_loader, val_loader, epochs=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


def acceptor():
    train_set = DataParser("train")
    F2I = train_set.get_F2I()
    dev_set = DataParser("dev", F2I)
    batch_size = 1
    hidden_dim = 64
    vocab_size = len(F2I)
    sequence_dim = 64
    embedding_dim = 5
    model = Model(vocab_size, embedding_dim, sequence_dim, hidden_dim, 1, batch_size)
    iterate_model(model, train_set.data_loader(batch_size), dev_set.data_loader(batch_size))


if __name__ == "__main__":
    acceptor()
