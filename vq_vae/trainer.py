import os
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.midi_tools import MidiMiniature
from utils.midi_dataset import MidiDataset


dir_path = os.path.dirname(os.path.abspath(__file__))


class TrainerVQVAE():
    def __init__(self, config_path, model):
        with open(config_path, "r", encoding='utf8') as fh:
            config = yaml.safe_load(fh)
        self.config = config["va_vae_trainer"]
        self.config_name = config["config_name"]
        assert self.config_name, "Please provide a config name in config.yaml file"
        self.model = model
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.epochs = self.config["epochs"]
        self.device = self.config["device"]
        self.dtype = torch.float32
        self.training_data = []
        self.validation_data = []
        self.training_loader = None
        self.validation_loader = None

    def augmentation(self, m, axis):
        if axis >= 0:
            m = np.flip(m, axis=axis)
        augmented = []
        try:
            up_limit = np.where(m)[0].min()
            down_limit = np.where(m)[0].max()
        except Exception:
            up_limit = 0
            down_limit = 0
        for i in range(1, up_limit+1):
            augmented.append(np.vstack((m[i:, :], np.zeros([i, 32]))))
        if down_limit:
            for i in range(1, m.shape[0] - down_limit):
                augmented.append(np.vstack((np.zeros([i, 32]), m[:-i, :])))
        return augmented

    def load_data(self, augmentation=False):
        # load midi data
        work_dir = os.path.join("data", "midi")
        miniaturizer = MidiMiniature(1)  # 1/4th
        tunes = [t for t in os.listdir(work_dir) if t.split(".")[1] == "mid"]
        to_validate_later = tunes.pop()
        # training
        for k, tune in enumerate(tunes):
            try:
                quarters = miniaturizer.make_miniature(os.path.join(work_dir, tune))
            except Exception as e:
                print(f"\nProblem processing {tune}:", e)
                continue
            quarters_to_extend = quarters.copy()
            if augmentation:
                for n in range(3):
                    axis = list(np.ones(len(quarters)).astype(int)*(n-1))
                    augmented_quarters = list(map(self.augmentation, quarters_to_extend, axis))
                    for a_q in augmented_quarters:
                        quarters.extend(a_q)
            self.training_data.extend(quarters)
            print(f"\rcreating training dataset. progress: {((k+1)/len(tunes)*100):.2f}%", end="")
        print("\ncreating validation dataset")
        # validation
        for tune in [to_validate_later]:
            quarters = miniaturizer.make_miniature(os.path.join(work_dir, tune))
            quarters_to_extend = quarters.copy()
            if augmentation:
                for n in range(3):
                    axis = list(np.ones(len(quarters)).astype(int) * (n-1))
                    augmented_quarters = list(map(self.augmentation, quarters_to_extend, axis))
                    for a_q in augmented_quarters:
                        quarters.extend(a_q)
            self.validation_data.extend(quarters)
        print(f"\n{len(self.training_data)} samples in training set")
        print(f"{len(self.validation_data)} samples in validation set\n")

    def get_train_loader(self):
        self.training_data = MidiDataset(self.training_data)
        self.validation_data = MidiDataset(self.validation_data)
        self.training_loader = DataLoader(self.training_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    pin_memory=True)
        self.validation_loader = DataLoader(self.validation_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    pin_memory=True)

    def vq_vae_loss(self, output, target):
        loss = nn.BCELoss(reduction='none')(output, target)
        return torch.mean(loss)

    def train(self):
        self.get_train_loader()
        state_dict_path = os.path.join(dir_path, self.config_name, "state_dict")
        if not os.path.exists(state_dict_path):
            os.mkdir(state_dict_path)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               amsgrad=False)
        self.model.train()
        train_res_recon_error = []
        train_res_perplexity = []
        valid_res_recon_error = []
        best_score = np.inf
        num_training_updates = self.epochs * self.training_data.__len__()

        for i in range(num_training_updates):
            self.model.train()
            data = next(iter(self.training_loader))
            data = data.to(self.dtype)
            data = data.to(self.device)
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = self.vq_vae_loss(data_recon, data)
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
            print(f"\repoch: {i//self.training_data.__len__()} | progress: {((i+1)%self.training_data.__len__())/self.training_data.__len__()*100:.2f}% | recon_error: {np.mean(train_res_recon_error[-100:]):-4f} | perplexity : {np.mean(train_res_perplexity[-100:]):.4f}", end='')

            if not i % self.training_data.__len__():
                print("\nValidation")
                with torch.no_grad():
                    self.model.eval()
                    for i_val in range(len(self.validation_data)):
                        valid_originals = next(iter(self.validation_loader)).to(torch.float32)
                        valid_originals = valid_originals.to(self.device)[0].unsqueeze(0)
                        vq_output_eval = self.model._pre_vq_conv(self.model._encoder(valid_originals))
                        _, valid_quantize, _, encodings = self.model._vq_vae(vq_output_eval)
                        valid_reconstructions = self.model._decoder(valid_quantize)
                        pred = np.round(valid_reconstructions.data.cpu().detach().numpy().squeeze())
                        true = np.round(valid_originals.data.cpu().detach().numpy().squeeze())
                        # we use SSE to evaluate reconstruction in validation
                        valid_recon_error = np.mean((true - pred)**2)
                        valid_res_recon_error.append(valid_recon_error)
                    print(f"validation recon_error: {np.mean(valid_res_recon_error):.4f}\n")
                    if np.mean(valid_res_recon_error) < best_score:
                        torch.save(self.model.state_dict(), os.path.join(state_dict_path, "best.pt"))
                        best_score = np.mean(valid_res_recon_error)
            torch.save(self.model.state_dict(), os.path.join(state_dict_path, "last.pt"))
