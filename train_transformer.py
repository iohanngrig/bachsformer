import numpy as np
from torch.utils.data import Dataset
import torch


class CodebooksDataset(Dataset):
    def __init__(self, device):
        codebooks = self.get_codebooks()
        self.data = np.array(codebooks)
        self.device = device
        self.length = len(codebooks[0])

    def get_codebooks(self):
        f = open("data/vocab/vocab_16_192length.txt", "r")
        codebooks_idx = []
        for line in f.readlines():
            line = [int(i) for i in (line.split("\n")[0].split(",")) if i]
            # line.insert(0,-1)
            codebooks_idx.append(line)
        return codebooks_idx

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        return 16

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length - 1

    def __getitem__(self, idx):
        # the inputs to the transformer will be the offset sequence
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y

    def shuffle_it(self):
        np.random.shuffle(self.data)


def batch_end_callback(trainer):
    print(f"\riter {trainer.iter_num}: train loss {trainer.loss.item():.5f}", end="")
    torch.save(trainer.model.state_dict(), "bachsformer")


if __name__ == "__main__":
    import sys, os
    from vq_vae.tools import LudovicoVAE
    from transformer_decoder_only.model import GPT
    from transformer_decoder_only.trainer import Trainer

    sys.path.append(os.getcwd())
    CONFIG = 'config.yaml'
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ludovico_vae = LudovicoVAE(config_path=CONFIG)
    print(f"*** Train Transformer with config: {ludovico_vae.config_name} ***")

    # get model
    try:
        model = ludovico_vae.get_model()
        print(f"Existing model {ludovico_vae.config_name} is restored")
    except:
        print(f"No model found with configuration: {ludovico_vae.config_name}")

    # get vocab
    ludovico_vae.codebooks2vocab(model)
    del model  # get rid of VQ-VAE, no longer needed
    train_dataset = CodebooksDataset(device)

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config["model_type"] = "gpt_bach"
    model_config["vocab_size"] = train_dataset.get_vocab_size()
    model_config["block_size"] = train_dataset.get_block_size()
    print(f"vocab_size: {model_config['vocab_size']}")
    print(f"block_size: {model_config['block_size']}")

    # model
    model = GPT(model_config).to(device)
    model_name = "bachsformer"

    # create a Trainer object
    trainer = Trainer(config_path=CONFIG, 
                      model=model, 
                      train_dataset=train_dataset)
    batch_size = trainer.config["batch_size"]
    steps_per_epoch = train_dataset.__len__() // batch_size
    trainer.config["max_iters"] = steps_per_epoch * trainer.config["multiplier"]
 
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Model loaded from pretrained {model_name}")
    except Exception as e:
        print(f"Model {model_name} is not pretrained")

    # train
    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()