# config.py

class Config(object):
    embed_size = 300
    hidden_size = 10	# default 10
    output_size = 10     # need to check matching the # of classes
    max_epochs = 100    # default 100
    lr = 0.5
    batch_size = 16
    networks = 1
    renyi_alpha = 0.0
    lr_step_size = 10
    lr_decay = 0.5
    patience_th = 100
    scale = 1.0
    ca = 0  # 1 indicates enable cosine annealing (max2, min0.5)
