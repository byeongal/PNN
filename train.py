import sys
import getopt

import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

import utils
import models

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else "cpu")

def train_model(model, epochs, train_loader, optimizer):
    model.train()
    loss_func = models.FocalLoss()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total = 0
        with tqdm(train_loader, desc='Train Epoch #{}'.format(epoch)) as t:
            for data, target in t:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                optimizer.zero_grad()
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                total += len(data)
                total_correct += output.round().eq(target).sum().item()
                total_loss += loss.item() * len(data)
                t.set_postfix(loss='{:.4f}'.format(total_loss / total), accuracy='{:.4f}'.format(total_correct / total))

def train(train_csv_path, model_path, batch_size, epochs, input_length, window_size):
    train_loader = utils.get_data_loader(train_csv_path, batch_size, True)
    model = models.PNN(input_length, window_size).to(DEVICE)
    model.apply(utils.init_normal)
    optimizer = optim.Adam(model.parameters())
    train_model(model, epochs, train_loader, optimizer)
    torch.save(model.state_dict(), model_path)

def main(argv):
    train_csv_path = None
    model_path = 'model.dat'
    batch_size = 128
    epochs = 10
    input_length = 1600
    window_size = 5
    optlist, args = getopt.getopt(argv[1:], '', ['help', 'train=', 'model=', 'batch_size=', 'epochs=', 'input_length=', 'window_size='])
    for opt, arg in optlist:
        if opt == '--help':
            utils.train_help()
            sys.exit(0)
        elif opt == '--train':
            train_csv_path = arg
        elif opt == '--model':
            model_path = arg
        elif opt == '--batch_size':
            batch_size = int(arg)
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt == '--input_length':
            input_length = int(arg)
        elif opt == '--window_size':
            window_size = int(arg)
    train(train_csv_path, model_path, batch_size, epochs, input_length, window_size)

if __name__ == '__main__':
    main(sys.argv)