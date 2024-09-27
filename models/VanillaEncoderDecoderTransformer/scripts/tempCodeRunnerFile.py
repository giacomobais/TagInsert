ept:
        model = make_model(len(word_to_idx), len(POS_to_idx), d_model = config['model']['d_model'], N=config['model']['n_heads'])
        model_opt = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = LambdaLR(optimizer=model_opt,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400),)
        trained_epochs = 0
        train_losses = []
        val_losses = []
        print('No pre-trained model found.')