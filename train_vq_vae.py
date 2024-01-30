from vq_vae.tools import LudovicoVAE, TrainerVQVAE     


if __name__=="__main__":
    config_name = "ludovico-mini"
    ludovico_vae = LudovicoVAE(config_name)
    print(f"*** Train VQ VAE with config: {config_name} ***")
    try:        
        # check if previous model exists
        model = ludovico_vae.get_model()
        print(f"Existing model {config_name} is restored")
    except:     
        # set model
        print(f"Setting the model: {config_name}")
        model = ludovico_vae.set_model()
    # train
    trainer = TrainerVQVAE(model, config_name, batch_size=512)
    # load data
    trainer.load_data(augmentation=False)
    # train 
    trainer.train()



     






