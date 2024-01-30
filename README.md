# bachsformer

This is a modified version of the original work by [Pierfrancesco Melucci](https://github.com/pier-maker92/bachsformer.git) adjusted to work on `CUDA` device and newer version of `PyTorch`. Main parameters of the model are enclosed into a `config.yaml` file. 

## Bach music generation

* Many fantastic attempts have been made to generate music through artificial intelligence. 
* Usually this task is addressed to processors, who obtain outstanding results in the generation of sequences, where the source vocabulary is constituted by the tokenization of the midi symbols. 
* This approach is characterized by the construction of the source vocabulary, as it is not midi symbols that constitute it, but is mediated by a mid-level representation. 
* First of all the original midi sequence is broken into quarters (1/4 of a musical bar). In doing so the time dimension is frozen. 
* Each slice is then reconstructed from a VQ-VAE, composed of a sequence of 16 codebooks, that I called Ludovico-VAE (to pay homage to the great Ludwig Van Beethoven). 
* Once the VQ-VAE has been trained, a decoder-only transformer (the Bachsformer) is trained on the sequences of codebooks that recompose the training-set.
* So the transformer learn how to generate coherent sequences of 192 codebooks indexes, which the Decoder of VQ-VAE will use to recompose the final midi score.
* You can think at is as variation-performer based on the chosen artist voucabolary (J.S.Bach in this case), which is discretize w.r.t. a subdivision of musical tempo.

* This repo comes with 2 pre-trained models, one for vq-vae and one for transformer. Please note that training is a very poor one, only few epochs for both the models.

* Listen to this example of generated output!


## Original Credits
* The implementation of transformer is taken by this awesome GPT implementation provided by @karpathy https://github.com/karpathy/minGPT
* Also thanks to @michelemancusi https://github.com/michelemancusi for its precious contribution to the idea!

## Installation

Clone the repo via 
```bash
git clone https://github.com/iohanngrig/bachsformer.git
```

Create a new conda environment (TODO: update bachsformer.yaml)
```bash
conda create -n bachsformer python=3.8 
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

You can generate via generate.py script (the weights are trained on `cuda` device))
```bash
python generate.py
```

## Dataset

The dataset provided for pre-trained models consist of 32 Golberg Variations from J.S. Bach. Midi files for training are placed inside data/midi folder. Feel free to try different/larger dataset but bear in mind that the midi files have to be perfectly quantized!

## Training

You have to train vq-vae first and then you will able to train the trasnformer on the codebooks indexes sequence
train vq-vae
```bash
python train_vq_vae.py
```
train bachsformer!
```bash
python train_transformer.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
For any question/feedback contact me at pierfrancesco.melucci@gmail.com

## License

MIT