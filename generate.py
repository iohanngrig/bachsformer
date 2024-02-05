import random
import argparse
from time import gmtime, strftime

import numpy as np
import torch

from utils.midi_tools import MidiMiniature
from vq_vae.model import LudovicoVAE
from transformer.model import GPT
from utils.common import set_seed

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("-t", "--temperature", type=float, default=0.9)
parser.add_argument("-tk", "--top_key", type=int, default=16)
parser.add_argument("-s", "--seed", type=int, default=42)


def generate_from_idx(model, idx, seed=None):
    """ model: vq_vae model, e.g. LudovicoVAE()
        idx: LongTensor of shape (b,t), see GPT model's forward method"""
    model.eval()
    num_embeddings = model._vq_vae._num_embeddings  # num_embeddings of the model or VectorQuantizer, e.g. 16
    embedding = model._vq_vae._embedding  # nn.Embedding(num_embeddings, embedding_dim), e.g. embedding_dim = 128
    if seed:
        set_seed(seed)
    encoding_indices = torch.tensor(idx, device=DEVICE).unsqueeze(1)
    encodings = torch.zeros(encoding_indices.shape[0], num_embeddings, device=DEVICE)
    encodings.scatter_(1, encoding_indices, 1)

    # Quantize and unflatten
    embedding_dim = model._vq_vae._embedding_dim
    quantized = torch.matmul(encodings, embedding.weight).view(torch.Size([1, 8, 2, embedding_dim]))  # last value is embedding_dim from model, e.g. 128
    quantized = quantized.detach().permute(0, 3, 1, 2).contiguous()
    return model._decoder(quantized)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    CONFIG = 'config.yaml'

    print("Using torch", torch.__version__)
    print("Device", DEVICE)

    args = parser.parse_args()
    set_seed(args.seed)

    ludovico_vae = LudovicoVAE(config_path=CONFIG)
    try:
        model = ludovico_vae.get_model()
    except Exception as e:
        print(f"Exception {e}. No model found with this configuration: {ludovico_vae.config_name}")

    # create a GPT instance
    gpt_model_config = GPT.get_default_config(config_path=CONFIG)

    # load model
    gpt_model = GPT(gpt_model_config).to(DEVICE)
    gpt_model_name = gpt_model.model_name

    try:
        gpt_model.load_state_dict(torch.load(gpt_model_name))
        print(f"Loaded pretrained model {gpt_model_name}")
    except Exception as e:
        print(f"Problem loading {gpt_model_name} model:", e)

    # generate codebooks index with transformers
    gpt_model.eval()
    if not args.file:
        first_idx = random.randint(0, 15)
        x = torch.tensor([first_idx], device=DEVICE).unsqueeze(0)
        generated = []
        for k in range(16):
            if not k:
                codebooks_idx = gpt_model.generate(x, 191, do_sample=True, top_k=args.top_key, temperature=args.temperature)
            else:
                codebooks_idx = gpt_model.generate(x, 176, do_sample=True, top_k=args.top_key, temperature=args.temperature,)
            codebooks_idx = codebooks_idx.data.cpu().detach().numpy().squeeze()
            x = torch.tensor(codebooks_idx[-16:], device=DEVICE).unsqueeze(0)
            print(x)
            quarters = np.array([codebooks_idx[i * 16: i * 16 + 16] for i in range(8)])
            for q in quarters:
                generated.append(q)
    else:
        file_name = args.file.split(".")[0]
        first_idx = f"from_input_{file_name}"
        x = torch.tensor(ludovico_vae.codebooks2vocab(model, tune_name=args.file), device=DEVICE).unsqueeze(0)
        generated = []
        for k in range(16):
            if not k:
                codebooks_idx = gpt_model.generate(x, 192 - x.shape[0], do_sample=True, top_k=args.top_key, temperature=args.temperature)
            else:
                codebooks_idx = gpt_model.generate(x, 176, do_sample=True, top_k=args.top_key, temperature=args.temperature)
            codebooks_idx = codebooks_idx.data.cpu().detach().numpy().squeeze()
            x = torch.tensor(codebooks_idx[-16:], device=DEVICE).unsqueeze(0)
            print(x)
            quarters = np.array([codebooks_idx[i*16:i*16+16] for i in range(8)])
            for q in quarters:
                generated.append(q)
    bars_generated = []
    for c in generated:
        new = np.round(generate_from_idx(model, c, seed=None).data.cpu().detach().numpy().squeeze())
        bars_generated.append(new)

    miniaturizer = MidiMiniature(1)  # 1/4th
    gen = miniaturizer.miniature2midi(bars_generated)
    temp = "_".join(str(args.temperature).split("."))
    now = strftime("%Y_%m_%d_%H_%M", gmtime())
    name = f"data/generated/gen_{first_idx}_temperature_{temp}_top_key_{args.top_key}_seed_{args.seed}_{now}.mid"
    gen.save(name)
    print(f"\nstored with name : {name}\n")
