# bachsformer

This is a modified version of the original work by [Pierfrancesco Melucci](https://github.com/pier-maker92/bachsformer.git) adjusted to work on `CUDA` device and newer version of `PyTorch` (2.2). Main parameters of the model are enclosed inside the `config.yaml` file. The model is trained on 86 works of J.S. Bach encapsulated withing the `data/midi` folder. The midi files are selected to be relatively "homogeneous". 

Notice that the generated audio is somewhat "fluent" sequence of musical "phrases" used by Bach in the provided midi files. The quality of generated audio strongly depends on the `seed` argument (given that the trained model is fixed). In some sense the model treats the music as a spoken text, which should not be surprising since the network is based on a GPT model.

This video is only trained on Bach's Goldberg Variations:

[![Watch the video](https://img.youtube.com/vi/eD-PVXEj9lI/hqdefault.jpg)](https://www.youtube.com/watch?v=eD-PVXEj9lI)


## Music Generation

* Many attempts have been made to generate music through artificial intelligence. 
* Usually this task is addressed to processors, who obtain impressive results in the generation of sequences, where the source vocabulary is constituted by the tokenization of the midi symbols. 
* This approach is characterized by the construction of the source vocabulary, as it is not midi symbols that constitute it, but is mediated by a mid-level representation. 
* First of all the original midi sequence is broken into quarters (1/4 of a musical bar). In doing so the time dimension is frozen. 
* The Vector-Quantized Variational Autoencoder (VQ-VAE) is a type of variational autoencoder where the autoencoder's encoder neural network emits discrete–not continuous–values by mapping the encoder's embedding values to a fixed number of codebook values.
* Each slice is then reconstructed from a VQ-VAE, composed of a sequence of 16 codebooks, that is called Ludovico-VAE (to pay homage to the great Ludwig Van Beethoven). 
The VQ-VAE was originally introduced in the Neural Discrete Representation Learning paper from Google.
* Once the VQ-VAE has been trained, a decoder-only transformer (the Bachsformer) is trained on the sequences of codebooks that recompose the training-set.
* So the transformer learns how to generate coherent sequences of 192 codebooks indexes, which the Decoder of VQ-VAE will use to recompose the final midi score.
* You can think at is as variation-performer based on the chosen artist vocabulary (J.S.Bach in this case), which is discretize w.r.t. a subdivision of musical tempo.
* This repo comes with 2 pre-trained models, one for vq-vae and one for transformer. 
* The midi output is generated inside the `data/generated` folder.

## Installation

Clone the repo via 
```bash
git clone https://github.com/iohanngrig/bachsformer.git
```

Create a conda environment from .yml file
```bash
conda env create -f bachsformer.yml
```

or 

Create a new conda environment
```bash
conda create -n bachsformer python=3.11 
```

Install PyTorch specific packages (select your version of `cuda`)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install the rest of the packages
```bash
pip install -r requirements.txt
```

## Generation

You can generate via generate.py script (the weights are trained on `cuda` device)
```bash
python generate.py
```

## Dataset

The extended dataset provided for pre-trained models consist of 86 works by J.S. Bach (instead of original 32 Goldberg Variations). Midi files for training are placed inside the `data/midi` folder. You can try different/larger datasets, however, note that the midi files have to be perfectly quantized!

## Training

You have to train vq-vae first and then you will able to train the transformer on the codebooks indexes sequence train vq-vae
```bash
python train_vq_vae.py
```
train bachsformer!
```bash
python train_transformer.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change. Please make sure to update tests as appropriate. For any question/feedback contact original [author](pierfrancesco.melucci@gmail.com) or current repository owner.

## Original Credits
* The implementation of transformer is taken by this awesome GPT implementation provided by [@karpathy](https://github.com/karpathy/minGPT)
* Also thanks to [@michelemancusi](https://github.com/michelemancusi) for its precious contribution to the idea!

## License

This project is licensed under the MIT license