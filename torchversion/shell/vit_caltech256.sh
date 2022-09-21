
python -m train_dml --data_name=caltech256 --network_name=vit:2 --epochs=30 --lr=0.001 --batch_size=64 --renyi_alpha=0.5 --network_pretrained --lr_decay=0.1 --lr_step_size=60 --validate --repeat=1 --weight_decay=0.0001 --grad_clip

python -m train_dml --data_name=caltech256 --network_name=vit:2 --epochs=30 --lr=0.001 --batch_size=64 --renyi_alpha=1.5 --network_pretrained --lr_decay=0.1 --lr_step_size=60 --validate --repeat=1 --weight_decay=0.0001 --grad_clip

python -m train_dml --data_name=caltech256 --network_name=vit:2 --epochs=30 --lr=0.001 --batch_size=64 --renyi_alpha=2 --network_pretrained --lr_decay=0.1 --lr_step_size=60 --validate --repeat=1 --weight_decay=0.0001 --grad_clip

python -m train_dml --data_name=caltech256 --network_name=vit:2 --epochs=30 --lr=0.001 --batch_size=64 --renyi_alpha=1 --network_pretrained --lr_decay=0.1 --lr_step_size=60 --validate --repeat=1 --weight_decay=0.0001 --grad_clip

python -m train_dml --data_name=caltech256 --network_name=vit:1 --epochs=30 --lr=0.001 --batch_size=64 --renyi_alpha=0 --network_pretrained --lr_decay=0.1 --lr_step_size=60 --validate --repeat=1 --weight_decay=0.0001 --grad_clip
