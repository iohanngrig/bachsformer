import os
import sys
from vq_vae.model import LudovicoVAE
from vq_vae.trainer import TrainerVQVAE


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(dir_path)
    CONFIG = os.path.join(dir_path, "config.yaml")

    ludovico_vae = LudovicoVAE(config_path=CONFIG)
    print(f"*** Train VQ VAE with config: {ludovico_vae.config_name} ***")
    try:
        # check if previous model exists
        model = ludovico_vae.get_model()
        print(f"Existing model {ludovico_vae.config_name} is restored")
    except Exception:
        # set model
        print(f"Setting the model: {ludovico_vae.config_name}")
        model = ludovico_vae.set_model()

    # instantiate trainer
    trainer = TrainerVQVAE(config_path=CONFIG, model=model)

    # load data
    trainer.load_data(augmentation=False)

    # train
    trainer.train()
