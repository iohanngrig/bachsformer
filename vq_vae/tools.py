import json
import yaml
import numpy as np
from tqdm import tqdm
from .vq_vae import Model
from .midi_tools import MidiMiniature
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class LudovicoVAE():
    def __init__(self, config_path):
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        self.config_name = config["config_name"]
        self.config = config[self.config_name]
    
        assert self.config_name, "Please inlcude config_name in a config.yaml file" 
        self.work_dir = os.path.join("vq_vae", self.config_name)
        self.device = self.config["device"]
        if not os.path.exists(self.work_dir):
            os.mkdir(self.work_dir)
    
    def set_model(self):
        num_hiddens = self.config["num_hiddens"]
        num_residual_hiddens = self.config["num_residual_hiddens"]
        num_residual_layers = self.config["num_residual_layers"]
        embedding_dim = self.config["embedding_dim"]
        num_embeddings = self.config["num_embeddings"]
        commitment_cost = self.config["commitment_cost"]
        decay = self.config["decay"]
        
        model = Model(num_hiddens, 
                      num_residual_layers, 
                      num_residual_hiddens,
                      num_embeddings, 
                      embedding_dim, 
                      commitment_cost, 
                      decay).to(self.device)
        config = {
            "num_hiddens":num_hiddens,
            "num_residual_layers":num_residual_layers,
            "num_residual_hiddens":num_residual_hiddens,
            "num_embeddings":num_embeddings,
            "embedding_dim":embedding_dim,
            "commitment_cost":commitment_cost,
            "decay":decay
            }
        with open(os.path.join(self.work_dir, f"{self.config_name}.json"), "w") as outfile:
            json.dump(config, outfile)
        return model
    
    def get_model(self, state_dict_name="last"):
        with open(os.path.join(self.work_dir, f"{self.config_name}.json")) as json_file:
            config = json.load(json_file)
        model = Model(config["num_hiddens"], 
                      config["num_residual_layers"], 
                      config["num_residual_hiddens"],
                      config["num_embeddings"],
                      config["embedding_dim"],
                      config["commitment_cost"],
                      config["decay"]).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.work_dir, f"state_dict/{state_dict_name}")))
        return model
    
    def codebooks2vocab(self, model, seq_len=192, tune_name=""):
        model.eval()
        work_dir = "data/midi/"
        miniaturizer = MidiMiniature(1) # 1/4th
        if not tune_name:
            tunes = [t for t in os.listdir(work_dir) if t.split(".")[1]=="mid"]
            f = open("data/vocab/vocab_16_192length.txt", "w")
        else:
            tunes = [tune_name]
        for tune in tqdm(tunes):
            quarters = miniaturizer.make_miniature(os.path.join(work_dir,tune))
            quarters = torch.tensor(np.reshape(np.array(quarters),(len(quarters),1,88,32))).to(torch.float32).to(self.device)
            
            vq_output_eval = model._pre_vq_conv(model._encoder(quarters))
            _, quantize, _, encodings = model._vq_vae(vq_output_eval)
            codebooks_idx = np.where(encodings.data.cpu().detach().numpy())[1]
            # 512 sequence in step of 16
            if tune_name:
                return codebooks_idx
            else: 
                slices = np.array([codebooks_idx[i*16:i*16+seq_len] for i in range(((codebooks_idx.shape[0]-seq_len)//16)+1)])
            for sequence in slices:
                sequence = ','.join(sequence.astype(str))
                f.write(f"{sequence}")
                f.write(f"\n")
        f.close()
           

class MidiDataset(Dataset):
    def __init__(self, quarters):
        self.data = np.reshape(np.array(quarters), 
                               (len(quarters), 1, quarters[0].shape[0], quarters[0].shape[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TrainerVQVAE():
    def __init__(self, config_path, model):
        with open(config_path) as fh:
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
        except:
            up_limit = 0; down_limit = 0
        for i in range(1, up_limit+1):
            augmented.append(np.vstack((m[i:, :], np.zeros([i, 32]))))
        if down_limit:
            for i in range(1, m.shape[0] - down_limit):
                augmented.append(np.vstack((np.zeros([i, 32]), m[:-i, :])))
        return augmented

    def load_data(self, augmentation=False):
        # load midi data
        work_dir = "data/midi/"
        miniaturizer = MidiMiniature(1) # 1/4th
        tunes = [t for t in os.listdir(work_dir) if t.split(".")[1]=="mid"]
        to_validate_later = tunes.pop()
        # training
        for k, tune in enumerate(tunes):
            quarters = miniaturizer.make_miniature(os.path.join(work_dir, tune))
            quarters_to_extend = quarters.copy()
            if augmentation:
                for n in range(3):
                    axis = list(np.ones(len(quarters)).astype(int)*(n-1))
                    augmented_quarters = list(map(self.augmentation,quarters_to_extend, axis))
                    for a_q in augmented_quarters: quarters.extend(a_q)
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
                    augmented_quarters = list(map(self.augmentation, quarters_to_extend,axis))
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
        if not os.path.exists(f"vq_vae/{self.config_name}/state_dict"):
            os.mkdir(f"vq_vae/{self.config_name}/state_dict")
        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.learning_rate, amsgrad=False)
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
            print(f"\repoch: {i//self.training_data.__len__()} | progress: {((i+1)%self.training_data.__len__())/self.training_data.__len__()*100:.2f}% | recon_error: {np.mean(train_res_recon_error[-100:]):-4f} | perplexity : {np.mean(train_res_perplexity[-100:]):.4f}",end='')
            
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
                        torch.save(self.model.state_dict(), f"vq_vae/{self.config_name}/state_dict/best")
                        best_score = np.mean(valid_res_recon_error)
            torch.save(self.model.state_dict(), f"vq_vae/{self.config_name}/state_dict/last")