import numpy as np
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, quarters):
        self.data = np.reshape(np.array(quarters),
                               (len(quarters), 1, quarters[0].shape[0], quarters[0].shape[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CodebooksDataset(Dataset):
    def __init__(self, device):
        codebooks = self.get_codebooks()
        self.data = np.array(codebooks)
        self.device = device
        self.length = len(codebooks[0])

    def get_codebooks(self):
        codebooks_idx = []
        with open("data/vocab/vocab_16_192length.txt", "r") as f:
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
