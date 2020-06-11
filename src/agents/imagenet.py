import os
import sys
import copy
import json
import dotmap
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

from src.utils.utils import \
    save_checkpoint as save_snapshot, \
    copy_checkpoint as copy_snapshot, \
    AverageMeter
from src.utils.setup import print_cuda_statistics
from src.utils.utils import l2_normalize
from src.utils.transforms import get_imagenet_transforms
from src.datasets.imagenet import ImageNet
from src.datasets.cifar10 import CIFAR10
from src.agents.base import BaseAgent
from src.models.resnet import resnet18
from src.models.logreg import LogisticRegression
from src.objectives.memory import MemoryBank
from src.objectives.instance import InstDisc, NCE, Ball, Ring


class ImageNetAgent(BaseAgent):

    def __init__(self, config):
        super(ImageNetAgent, self).__init__(config)
        # initialize objects specific to local aggregation
        self.init_memory_bank()

        self.val_acc = []
        self.train_loss = []
        self.train_extra = []

    def init_memory_bank(self, attr_name='memory_bank'):
        data_len = len(self.train_dataset)
        memory_bank = MemoryBank(data_len, self.config.model_params.out_dim, self.device)
        setattr(self, attr_name, memory_bank)

    def load_memory_bank(self, memory_bank, attr_name='memory_bank'):
        memory_bank = copy.deepcopy(memory_bank)
        memory_bank.device = self.device
        memory_bank._bank = memory_bank._bank.cpu()
        memory_bank._bank = memory_bank._bank.to(self.device)
        setattr(self, attr_name, memory_bank)

    def load_datasets(self):
        train_transforms, test_transforms = get_imagenet_transforms(
            self.config.data_params.image_size, 256)

        self.train_dataset = ImageNet(train=True, image_transforms=train_transforms)
        self.val_dataset = ImageNet(train=False, image_transforms=test_transforms)

        train_samples = self.train_dataset.dataset.samples
        train_labels = [train_samples[i][1] for i in range(len(train_samples))]
        self.train_ordered_labels = np.array(train_labels)

    def create_model(self):
        assert self.config.data_params.image_size == 224
        resnet_class = getattr(models, self.config.model_params.resnet_version)
        model = resnet_class(pretrained=False,
                             num_classes=self.config.model_params.out_dim)
        self.model = model.to(self.device)

    def create_optimizer(self):
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.optim_params.learning_rate,
                                     momentum=self.config.optim_params.momentum,
                                     weight_decay=self.config.optim_params.weight_decay)

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()
        epoch_loss = AverageMeter()

        for batch_i, (indices, images, _) in enumerate(self.train_loader):
            batch_size = images.size(0)

            indices = indices.to(self.device)
            images = images.to(self.device)

            outputs = self.model(images)
            loss_class = globals()[self.config.loss_params.loss]
            loss_fn = loss_class(indices, outputs, self.memory_bank,
                                 k=self.config.loss_params.k,
                                 t=self.config.loss_params.t,
                                 m=self.config.loss_params.m,
                                 nsp_back=self.config.loss_params.nsp_back,
                                 nsp_close=self.config.loss_params.nsp_close)
            loss = loss_fn.get_loss()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                new_data_memory = loss_fn.updated_new_data_memory()
                self.memory_bank.update(indices, new_data_memory)

            epoch_loss.update(loss.item(), batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})

            self.train_loss.append(epoch_loss.val)
            self.train_extra.append(extra)

            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def validate(self):
        num_batches = self.val_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self.model.eval()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for _, images, labels in self.val_loader:
                batch_size = images.size(0)

                images = images.to(self.device)
                outputs = self.model(images)

                all_dps = self.memory_bank.get_all_dot_products(outputs)
                _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
                neighbor_idxs = neighbor_idxs.squeeze(1)  # shape: batch_size
                neighbor_idxs = neighbor_idxs.cpu().numpy()  # convert to numpy
                neighbor_labels = self.train_ordered_labels[neighbor_idxs]
                neighbor_labels = torch.from_numpy(neighbor_labels).long()

                num_correct += torch.sum(neighbor_labels == labels).item()
                num_total += batch_size

                tqdm_batch.set_postfix({"Val Accuracy": num_correct / num_total})
                tqdm_batch.update()

        self.current_val_iteration += 1
        self.current_val_metric = num_correct / num_total

        if self.current_val_metric >= self.best_val_metric:  # NOTE: >= for accuracy
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0   # reset patience
        else:
            self.iter_with_no_improv += 1  # no improvement

        tqdm_batch.close()

        self.val_acc.append(self.current_val_metric)
        return self.current_val_metric

    def load_checkpoint(self, filename, checkpoint_dir=None,
                        load_memory_bank=True, load_model=True,
                        load_optim=False, load_epoch=False):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']
                self.val_acc = list(checkpoint['val_acc'])
                self.train_loss = list(checkpoint['train_loss'])
                self.current_val_metric = checkpoint['val_metric']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            if load_memory_bank: # load memory_bank
                self._load_memory_bank(checkpoint['memory_bank'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if self.current_epoch % self.config.copy_checkpoint_freq == 0:
            copy_snapshot(
                filename=filename, folder=self.config.checkpoint_dir,
                copyname='checkpoint_epoch{}.pth.tar'.format(self.current_epoch),
            )

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'memory_bank': self.memory_bank,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'val_acc': np.array(self.val_acc),
            'train_loss': np.array(self.train_loss),
        }
        is_best = ((self.current_val_metric == self.best_val_metric) or
                   not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)
        self.copy_checkpoint()


class ImageNetTransferAgent(BaseAgent):
    def __init__(self, config):
        super(ImageNetTransferAgent, self).__init__(config)

        agent = self.load_trained_agent(config.trained_agent_exp_dir,
                                        config.gpu_device,
                                        checkpoint_name=config.checkpoint_name)
        if hasattr(agent, 'model'):
            self.resnet = copy.deepcopy(agent.model)
            self.resnet.load_state_dict(agent.model.state_dict())
        elif hasattr(agent, 'img_model'):
            self.resnet = copy.deepcopy(agent.img_model)
            self.resnet.load_state_dict(agent.img_model.state_dict())
        else:
            raise Exception('is this an image module?')

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.resnet = self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.val_acc = []
        self.val_acc_top5 = []
        self.train_loss = []

    def load_trained_agent(self, exp_dir, gpu_device, checkpoint_name='model_best.pth.tar'):
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
        config = checkpoint['config']
        # overwrite GPU since we might want to use a different GPU
        config.gpu_device = gpu_device

        self.logger.info("Loading trained Agent from filesystem")
        AgentClass = globals()[config.agent]
        agent = AgentClass(config)
        agent.load_checkpoint(
            checkpoint_name, 
            checkpoint_dir=checkpoint_dir, 
            load_memory_bank=True, 
            load_model=True,
            load_optim=False,
            load_epoch=False,
        )
        agent.model.eval()
        return agent

    def load_datasets(self):
        train_transforms, test_transforms = get_imagenet_transforms(
            self.config.data_params.image_size, 256)
        train_dataset = ImageNet(train=True, image_transforms=train_transforms)
        val_dataset = ImageNet(train=False, image_transforms=test_transforms)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def create_model(self):
        assert self.config.data_params.image_size == 224
        model = LogisticRegression(512*7*7, 1000)
        model = model.to(self.device)
        self.model = model

    def create_optimizer(self):
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.config.optim_params.learning_rate,
                                     momentum=self.config.optim_params.momentum,
                                     weight_decay=self.config.optim_params.weight_decay)

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()  # turn on train mode
        epoch_loss = AverageMeter()

        for batch_i, (_, images, labels) in enumerate(self.train_loader):
            batch_size = images.size(0)
            images = images.to(self.device).float()
            labels = labels.to(self.device)

            with torch.no_grad():
                embeddings = self.resnet(images)
                embeddings = embeddings.view(batch_size, -1)

            logits = self.model(embeddings)
            loss = F.cross_entropy(logits, labels)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss.update(loss.item(), batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})

            self.train_loss.append(epoch_loss.val)
            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def validate(self):
        num_batches = self.val_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self.model.eval()
        num_correct = 0.
        num_correct_top5 = 0.
        num_total = 0.

        with torch.no_grad():
            for _, images, labels in self.val_loader:
                batch_size = images.size(0)
                images = images.to(self.device)

                embeddings = self._forward_func(self.resnet, images)
                embeddings = embeddings.view(batch_size, -1)
                logits = self.model(embeddings)

                preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
                preds = preds.long().cpu()

                orders = torch.argsort(F.log_softmax(logits, dim=1), dim=1, descending=True)[:, :5]
                orders = orders.long().cpu()
                
                for j in range(batch_size):
                    in_top5 = float(labels[j].item() in orders[j].numpy())
                    num_correct_top5 += in_top5

                num_correct += torch.sum(preds == labels).item()
                num_total += batch_size

                tqdm_batch.set_postfix({
                    "Val Top1": num_correct / num_total,
                    "Val Top5": num_correct_top5 / num_total,
                })
                tqdm_batch.update()

        self.current_val_iteration += 1
        self.current_val_metric = num_correct / num_total

        if self.current_val_metric >= self.best_val_metric:  # NOTE: >= for accuracy
            self.best_val_metric = self.current_val_metric
            self.iter_with_no_improv = 0   # reset patience
        else:
            self.iter_with_no_improv += 1  # no improvement

        tqdm_batch.close()

        self.val_acc.append(self.current_val_metric)
        self.val_acc_top5.append(num_correct_top5 / num_total)

        return self.current_val_metric

    def load_checkpoint(self, filename, checkpoint_dir=None,
                        load_model=True, load_optim=False, load_epoch=False):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']
                self.val_acc = list(checkpoint['val_acc'])
                self.val_acc_top5 = list(checkpoint['val_acc_top5'])
                self.train_loss = list(checkpoint['train_loss'])

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'train_loss': np.array(self.train_loss),
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'val_acc': np.array(self.val_acc),
            'val_acc_top5': np.array(self.val_acc_top5),
            'config': self.config,
        }

        is_best = ((self.current_val_metric == self.best_val_metric) or
                   not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)
        self.copy_checkpoint()


class CIFAR10Agent(ImageNetAgent):

    def load_datasets(self):
        train_transforms, test_transforms = get_imagenet_transforms(
            self.config.data_params.image_size, 256)
        train_dataset = CIFAR10(train=True, image_transforms=train_transforms)
        val_dataset = CIFAR10(train=False, image_transforms=test_transforms)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_labels = train_dataset.dataset.targets
        self.train_ordered_labels = np.array(train_labels)


class CIFAR10TransferAgent(ImageNetTransferAgent):

    def _load_datasets(self):
        train_transforms, test_transforms = self._load_image_transforms()
        train_dataset = CIFAR10(train=True, image_transforms=train_transforms)
        val_dataset = CIFAR10(train=False, image_transforms=test_transforms)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def create_model(self):
        model = LogisticRegression(512*7*7, 10)
        model = model.to(self.device)
        self.model = model
