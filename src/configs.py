class CFG:
    # debug settings
    debug = False
    
    # general settings
    seed = 42
    train = True
    
    # model settings
    model = "microsoft/deberta-v3-base"
    gradient_checkpointing = True
    
    # training settings
    epochs = 4
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    
    # optimizer settings
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    
    # scheduler settings
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    
    # data loading settings
    num_workers = 4
    batch_size = 8
    max_len = 512
    
    # training process settings
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    print_freq = 20
    
    # other settings
    apex = True
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    
    # debug mode adjustments
    if debug:
        epochs = 2
        trn_fold = [0]