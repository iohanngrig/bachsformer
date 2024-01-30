"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import sys
import os
import time
import yaml
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.getcwd())
CONFIG = 'transformer_decoder_only/config.yaml'

class Trainer:
    @staticmethod
    def get_default_config(config_path=CONFIG):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config["trainer"]

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config["device"]
        self.model = self.model.to(self.device)
        print("Running on device:", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    
    def trainloader_setup(self, config):
        self.train_dataset.shuffle_it()
        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=self.train_dataset.__len__()),
            shuffle=False,
            pin_memory=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        return train_loader

    def run(self):
        model, config = self.model, self.config
        train_loader = self.trainloader_setup(config)
        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                train_loader = self.trainloader_setup(config)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_norm_clip"])
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config["max_iters"] is not None and self.iter_num >= config["max_iters"]:
                break
