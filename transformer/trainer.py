"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import time
import yaml
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config_path, model, train_dataset):
        with open(config_path, "r", encoding='utf8') as fh:
            config = yaml.safe_load(fh)
        self.config = config["trainer"]
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config["device"]
        self.model = self.model.to(self.device)
        print("Running on device:", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.loss = None
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

    def trainloader_setup(self):
        self.train_dataset.shuffle_it()
        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=len(self.train_dataset)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )
        return train_loader

    def run(self):
        self.model.train()
        train_loader = self.trainloader_setup()

        # setup the optimizer
        self.optimizer = self.model.configure_optimizers(self.config)
        self.iter_num = 0

        writer = SummaryWriter('runs/log_transformer')

        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                train_loader = self.trainloader_setup()
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = self.model(x, y)
            
            writer.add_scalar('training_loss', self.loss.cpu().item(), global_step=self.iter_num+1)

            # backprop and update the parameters
            self.optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_norm_clip"])
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1

            # termination conditions
            if self.config["max_iters"] is not None and self.iter_num >= self.config["max_iters"]:
                writer.close()
                break
