import torch
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_batch(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        acc = (torch.eq(y_pred.argmax(dim=-1), y)).sum() / x.shape[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    def train_one_epoch(self, train_loader):
        self.model.train()
        loss_inner_epoch = []
        acc_inner_epoch = []
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            loss, acc = self.train_one_batch(x, y)
            loss_inner_epoch.append(loss)
            acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()

    def eval_one_batch(self, x, y):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        acc = (torch.eq(y_pred.argmax(dim=-1), y)).sum() / x.shape[0]
        return loss.item(), acc.item()

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        loss_inner_epoch = []
        acc_inner_epoch = []
        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            loss, acc = self.eval_one_batch(x, y)
            loss_inner_epoch.append(loss)
            acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()

    def test(self, test_loader):
        self.model.eval()
        loss_inner_epoch = []
        acc_inner_epoch = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                loss, acc = self.eval_one_batch(x, y)
                loss_inner_epoch.append(loss)
                acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()


