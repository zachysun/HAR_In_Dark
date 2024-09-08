import torch
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_batch(self, x_rgb, x_flow, labels):
        self.optimizer.zero_grad()
        y_pred = self.model(x_rgb, x_flow)
        loss = self.criterion(y_pred, labels)
        acc = (torch.eq(y_pred.argmax(dim=-1), labels)).sum() / x_rgb.shape[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    def train_one_epoch(self, train_loader_rgb, train_loader_flow):
        self.model.train()
        loss_inner_epoch = []
        acc_inner_epoch = []
        for (x_rgb, labels), (x_flow, _) in zip(train_loader_rgb, train_loader_flow):
            x_rgb = x_rgb.to(self.device)
            x_flow = x_flow.to(self.device)
            labels = labels.to(self.device)
            loss, acc = self.train_one_batch(x_rgb, x_flow, labels)
            loss_inner_epoch.append(loss)
            acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()

    def eval_one_batch(self, x_rgb, x_flow, labels):
        y_pred = self.model(x_rgb, x_flow)
        loss = self.criterion(y_pred, labels)
        acc = (torch.eq(y_pred.argmax(dim=-1), labels)).sum() / labels.shape[0]
        return loss.item(), acc.item()

    def eval_one_epoch(self, val_loader_rgb, val_loader_flow):
        self.model.eval()
        loss_inner_epoch = []
        acc_inner_epoch = []
        for (x_rgb, labels), (x_flow, _) in zip(val_loader_rgb, val_loader_flow):
            x_rgb = x_rgb.to(self.device)
            x_flow = x_flow.to(self.device)
            labels = labels.to(self.device)
            loss, acc = self.eval_one_batch(x_rgb, x_flow, labels)
            loss_inner_epoch.append(loss)
            acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()

    def test(self, test_loader_rgb, test_loader_flow):
        self.model.eval()
        loss_inner_epoch = []
        acc_inner_epoch = []
        with torch.no_grad():
            for (x_rgb, labels), (x_flow, _) in zip(test_loader_rgb, test_loader_flow):
                x_rgb = x_rgb.to(self.device)
                x_flow = x_flow.to(self.device)
                labels = labels.to(self.device)
                loss, acc = self.eval_one_batch(x_rgb, x_flow, labels)
                loss_inner_epoch.append(loss)
                acc_inner_epoch.append(acc)
        return np.array(loss_inner_epoch).mean(), np.array(acc_inner_epoch).mean()


