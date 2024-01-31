from vq_vae.tools import LudovicoVAE, TrainerVQVAE     


if __name__=="__main__":
    import sys, os
    sys.path.append(os.getcwd())
    CONFIG = 'config.yaml'

    ludovico_vae = LudovicoVAE(config_path=CONFIG)
    print(f"*** Train VQ VAE with config: {ludovico_vae.config_name} ***")
    try:        
        # check if previous model exists
        model = ludovico_vae.get_model()
        print(f"Existing model {ludovico_vae.config_name} is restored")
    except:     
        # set model
        print(f"Setting the model: {ludovico_vae.config_name}")
        model = ludovico_vae.set_model()
    # train
    trainer = TrainerVQVAE(config_path=CONFIG,
                           model=model)
    # load data
    trainer.load_data(augmentation=False)
    # train 
    trainer.train()



     






