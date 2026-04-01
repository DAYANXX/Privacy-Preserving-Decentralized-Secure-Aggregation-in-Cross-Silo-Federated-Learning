#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import time


def encrypt_data(data, key, iv):
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return encrypted_data


def generate_shared_keys(num_users):
    keys = {}
    for i in range(num_users):
        for j in range(i + 1, num_users):
            keys[(i, j)] = os.urandom(32)
            keys[(j, i)] = keys[(i, j)]
    return keys


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def secure_train(self, net, round_number, keys, num_users, idx):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print(
                        f'Update Epoch: {iter}/{batch_idx * len(images)}/{len(self.ldr_train.dataset)} ({100. * batch_idx / len(self.ldr_train):.0f}%) Loss: {loss.item():.6f}')
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        state_dict = net.state_dict()

        total_masks = {}
        for name, param in state_dict.items():
            shape = param.shape
            num_bytes = param.numel()

            total_mask = torch.zeros(param.size(), dtype=torch.float32)

            for j in range(num_users):
                if j == idx:
                    continue
                key = keys[idx, j]

                cipher = Cipher(algorithms.AES(key), modes.CTR(round_number.to_bytes(16, byteorder='big', signed=True)),
                                backend=default_backend())
                encryptor = cipher.encryptor()
                random_bytes = encryptor.update(b'\x00' * num_bytes) + encryptor.finalize()

                random_floats = np.frombuffer(random_bytes, dtype=np.uint8) / 255.0
                random_tensor = torch.tensor(random_floats, dtype=torch.float32).view(shape)

                if j < idx:
                    total_mask -= random_tensor
                else:
                    total_mask += random_tensor

            total_masks[name] = total_mask

        masked_state_dict = {name: param + total_masks[name] for name, param in state_dict.items()}

        return masked_state_dict, sum(epoch_loss) / len(epoch_loss)
