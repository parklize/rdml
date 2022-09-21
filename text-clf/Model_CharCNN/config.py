# config.py

class Config(object):
    num_channels = 256
    linear_size = 256
    output_size = 5 # set according to dataset
    max_epochs = 20
    lr = 0.0001
    batch_size = 128 # default 128
    seq_len = 300 # 1014 in original paper
    dropout_keep = 0.5
    networks = 1 
    renyi_alpha = 0.0 
    lr_step_size = 3
    lr_decay = 0.9 
    patience_th = 100 
    scale = 1.0 
    ca = 0  # 1 indicates enable cosine annealing (max2, min0.5)
    eval_val_every100 = 0 # Set to 0 for speeding up, 0 for diagnose  
