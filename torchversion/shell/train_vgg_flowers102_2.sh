python -m train_dml --data_name=flowers102 --network_name=resnet32:2 --epochs=200 --lr=0.1 --batch_size=64 --renyi_alpha=2 --nesterov --network_pretrained --lr_decay=0.1 --lr_step_size=60 --repeat=5

python -m train_dml --data_name=flowers102 --network_name=resnet32:2 --epochs=200 --lr=0.1 --batch_size=64 --renyi_alpha=0.5 --nesterov --network_pretrained --lr_decay=0.1 --lr_step_size=60 --repeat=5
