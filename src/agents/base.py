import os
import sys
import copy
import json
import dotmap
import pickle
import logging
import numpy as np
from tqdm import tqdm
from itertools import product, chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

from src.utils.utils import save_checkpoint as save_snapshot, \
                            copy_checkpoint as copy_snapshot
from src.utils.setup import print_cuda_statistics


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

        self.set_seed()  # set seed as early as possible

        self.load_datasets()
        self.train_loader, self.train_len = self.create_dataloader(
            self.train_dataset,
            shuffle=True,
        )
        self.val_loader, self.val_len = self.create_dataloader(
            self.val_dataset,
            shuffle=False,
        )

        self.choose_device()
        self.create_model()
        self.create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_metric = 0
        self.best_val_metric = 0
        self.iter_with_no_improv = 0

    def set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda: torch.cuda.manual_seed(self.manual_seed)

        if self.cuda:
            if not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]

            # NOTE: we do not support multi-gpu run for now
            gpu_device = self.config.gpu_device[0]
            self.logger.info("User specified 1 GPU: {}".format(gpu_device))
            self.device = torch.device("cuda")
            torch.cuda.set_device(gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def load_datasets(self):
        raise NotImplementedError

    def create_dataloader(self, dataset, shuffle=True):
        dataset_size = len(dataset)
        print('LENGTH OF DATASET: ', dataset_size)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size,
                            shuffle=shuffle, pin_memory=True,
                            num_workers=self.config.data_loader_workers)

        return loader, dataset_size

    def create_model(self):
        raise NotImplementedError

    def create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run_validation(self):
        self.validate()

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            if (self.config.validate and
                epoch % self.config.optim_params.validate_freq == 0):
                self.validate()  # validate every now and then
            self.save_checkpoint()

            # check if we should quit early bc bad perf
            if self.iter_with_no_improv > self.config.optim_params.patience:
                self.logger.info("Exceeded patience. Stop training...")
                break

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')
