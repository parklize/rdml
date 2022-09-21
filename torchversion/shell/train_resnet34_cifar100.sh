# python -m train_dml --data_name=cifar100 --network_name=resnet34:1 --epochs=200 --lr=0.1 --batch_size=128 --renyi_alpha=0 --nesterov --network_pretrained --lr_decay=0.2 --lr_step_size=60 --validate --repeat=1

# python -m train_dml --data_name=cifar100 --network_name=resnet34:2 --epochs=200 --lr=0.1 --batch_size=128 --renyi_alpha=1 --nesterov --network_pretrained --lr_decay=0.2 --lr_step_size=60 --validate --repeat=1

python -m train_dml --data_name=cifar100 --network_name=resnet34:2 --epochs=200 --lr=0.1 --batch_size=128 --renyi_alpha=2 --nesterov --network_pretrained --lr_decay=0.2 --lr_step_size=60 --validate --repeat=1
