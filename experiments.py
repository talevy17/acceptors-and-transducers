from DataUtils import ExperimentsParser as parser
from SimpleLstm import SimpleLstm

import time
import torch
import torch.nn as nn

from gen_examples import generate_train_dev_test


def calc_batch_accuracy(predictions, labels):
    correct = wrong = 0
    for pred, label in zip(predictions, labels):
        if int(torch.argmax(pred)) == int(label):
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def train(model, loader, optimizer, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    data, labels = loader
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for sequence, label in zip(data, labels):
        optimizer.zero_grad()
        predictions = model(sequence.squeeze(1))
        label = torch.tensor([label])
        predictions = predictions.view(1, -1)
        loss = criterion(predictions, label)
        acc = calc_batch_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / len(data), epoch_acc / len(data), model


def evaluate(model, loader, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    data, labels = loader
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    model.eval()
    with torch.no_grad():
        for sequence, label in zip(data, labels):
            predictions = model(sequence.squeeze(1))
            label = torch.tensor([label])
            predictions = predictions.view(1, -1)
            loss = criterion(predictions, label)
            acc = calc_batch_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return epoch_loss / len(data), epoch_acc / len(data)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def iterate_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, model = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


def acceptor():
    train_set = parser("train")
    F2I = train_set.get_F2I()
    dev_set = parser("dev", F2I)
    batch_size = 1
    hidden_dim = 100
    sequence_dim = 50
    embedding_dim = len(F2I)
    model = SimpleLstm(embedding_dim, sequence_dim, hidden_dim, 2, batch_size)
    data, labels = train_set.encoder(), train_set.get_labels()
    dev_data, dev_labels = dev_set.encoder(), dev_set.get_labels()
    iterate_model(model, (data, labels), (dev_data, dev_labels))


if __name__ == "__main__":
    positive_regex = r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
    negative_regex = r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'
    regexes = [negative_regex, positive_regex]
    generate_train_dev_test(3000, 'dev', regexes, is_test=False)
    generate_train_dev_test(15000, 'train', regexes, is_test=False)
    acceptor()
