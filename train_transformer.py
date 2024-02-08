import os
import yaml
import torch
from utils.midi_dataset import CodebooksDataset


dir_path = os.path.dirname(os.path.realpath(__file__))
CONFIG = os.path.join(dir_path, 'config.yaml')

with open(CONFIG, "r", encoding='utf8') as fh:
    config = yaml.safe_load(fh)

MODEL_TYPE = config["model_type"]  # "gpt_bach2"
MODEL_NAME = config["model_name"]  # "bachsformer.pth"

MODEL_PATH = os.path.join(dir_path, MODEL_NAME)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def batch_end_callback(trainer):
    """ Saves the model weighs when this callback is triggered.
        Note: Just using torch.save will return immediately, fh will be closed
        and the thread will be unable to write to the file. Using "with"
        context manager helps to resolve this issue, ensuring that the file
        stream process doesn't block other processes."""
    print(f"\riter {trainer.iter_num}: train loss {trainer.loss.item():.5f}", end="")
    with open(MODEL_PATH, 'wb') as fh:
        torch.save(trainer.model.state_dict(), fh)


if __name__ == "__main__":
    from vq_vae.model import LudovicoVAE
    from transformer.model import GPT
    from transformer.trainer import Trainer

    ludovico_vae = LudovicoVAE(config_path=CONFIG)
    print(f"*** Train Transformer with config: {ludovico_vae.config_name} ***")

    # get model
    try:
        model = ludovico_vae.get_model()
        print(f"Existing model {ludovico_vae.config_name} is restored")
    except Exception:
        print(f"No model found with configuration: {ludovico_vae.config_name}")
        raise

    # get vocab
    ludovico_vae.codebooks2vocab(model)
    del model  # get rid of VQ-VAE, no longer needed

    train_dataset = CodebooksDataset(device=DEVICE)

    # create a GPT instance
    model_config = GPT.get_default_config(config_path=CONFIG)
    model_config["model_type"] = MODEL_TYPE
    model_config["vocab_size"] = train_dataset.get_vocab_size()
    model_config["block_size"] = train_dataset.get_block_size()
    print(f"vocab_size: {model_config['vocab_size']}")
    print(f"block_size: {model_config['block_size']}")

    # model
    model = GPT(model_config).to(DEVICE)

    # create a Trainer object
    trainer = Trainer(config_path=CONFIG, model=model, train_dataset=train_dataset)
    batch_size = trainer.config["batch_size"]
    steps_per_epoch = len(train_dataset) // batch_size
    trainer.config["max_iters"] = steps_per_epoch * trainer.config["multiplier"]

    try:
        model.load_state_dict(torch.load(MODEL_NAME))
        print(f"Model loaded from pretrained {MODEL_NAME}")
    except Exception:
        print(f"Model {MODEL_NAME} is not pretrained")

    # train
    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()
