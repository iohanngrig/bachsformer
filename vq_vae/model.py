import os
import json
import yaml
import numpy as np
from tqdm import tqdm

import torch

from utils.midi_tools import MidiMiniature
from .vq_vae import Model

dir_path = os.path.dirname(os.path.abspath(__file__))


class LudovicoVAE():
    def __init__(self, config_path):
        with open(config_path, "r", encoding='utf8') as fh:
            config = yaml.safe_load(fh)
        self.config_name = config["config_name"]
        self.config = config[self.config_name]

        assert self.config_name, "Please inlcude config_name in a config.yaml file"
        self.work_dir = os.path.join(dir_path, self.config_name)
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
            "num_hiddens": num_hiddens,
            "num_residual_layers": num_residual_layers,
            "num_residual_hiddens": num_residual_hiddens,
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "commitment_cost": commitment_cost,
            "decay": decay
            }
        with open(os.path.join(self.work_dir, f"{self.config_name}.json"), "w") as outfile:
            json.dump(config, outfile)
        return model

    def get_model(self, state_dict_name="last.pt"):
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

    def codebooks2vocab(self, model, seq_len=192, tune_name=None):
        model.eval()
        work_dir = os.path.join("data", "midi")
        vocab_file_path = os.path.join("data", "vocab", "vocab_16_192length.txt")
        miniaturizer = MidiMiniature(1)  # 1/4th
        if not tune_name:
            tunes = [t for t in os.listdir(work_dir) if t.split(".")[1] == "mid"]
            f = open(vocab_file_path, "w")
        else:
            tunes = [tune_name]
        for tune in tqdm(tunes):
            quarters = miniaturizer.make_miniature(os.path.join(work_dir, tune))
            quarters = torch.tensor(np.reshape(np.array(quarters),
                                               (len(quarters), 1, 88, 32))).to(torch.float32).to(self.device)
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
                f.write("\n")
        f.close()
