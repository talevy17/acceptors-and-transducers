import time
import torch
import torch.nn as nn
from DataUtils import NONE
import numpy as np
import csv
import math
import copy


def calc_batch_accuracy(predictions, labels, I2L):
    total = 0
    for pred, label in zip(predictions, labels):
        correct = wrong = 0
        for word_pred, word_label in zip(pred, label):
            string_label = I2L[int(word_label)]
            if string_label == NONE:
                continue
            if int(torch.argmax(word_pred)) == int(word_label):
                if not string_label == 'O':
                    correct += 1
            else:
                wrong += 1
        total += correct / (correct + wrong)
    return total / len(predictions)


def train(model, loader, optimizer, criterion, epoch, I2L):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for sequence, label in loader:
        optimizer.zero_grad()
        predictions = model(sequence.squeeze(1))
        loss = criterion(predictions.permute(1, 2, 0), label)
        acc = calc_batch_accuracy(predictions.permute(1, 0, 2), label, I2L)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / len(loader), epoch_acc / len(loader), model


def evaluate(model, loader, criterion, epoch, I2L, mode, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    export_tracker = math.floor(500 / batch_size)
    init_export_tracker = copy.deepcopy(export_tracker)
    file = open('./Data/results/{0}.csv'.format(mode), 'w+')
    writer = csv.writer(file)
    writer.writerow(["epoch", "sentences", "loss", "accuracy"])
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    model.eval()
    with torch.no_grad():
        for index, (sequence, label) in enumerate(loader):
            predictions = model(sequence.squeeze(1))
            loss = criterion(predictions.permute(1, 2, 0), label)
            acc = calc_batch_accuracy(predictions.permute(1, 0, 2), label, I2L)
            epoch_loss += loss.item()
            epoch_acc += acc
            if index == export_tracker:
                writer.writerow([str(epoch), str(export_tracker * batch_size), str(epoch_loss / index), str(epoch_acc / index)])
                export_tracker += init_export_tracker
    file.close()
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def iterate_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, I2L={},
                  ignore_index=np.inf, mode='a-pos', batch_size=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index) if not ignore_index == np.inf else nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, model = train(model, train_loader, optimizer, criterion, epoch, I2L)
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch, I2L, mode, batch_size)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')
