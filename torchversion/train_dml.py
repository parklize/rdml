import argparse
import json
import logging
import numpy as np
import pandas as pd
import sys
import torch

from pathlib import Path
from pprint import pprint
from PIL import Image
# from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from torchmetrics import F1Score, Accuracy
from traceback import print_exc
from typing import Union

from core.dml import DML
from dataloader import preprocess_data
from model import get_network

np.random.seed(250)
torch.manual_seed(250)
logging.getLogger().setLevel(logging.INFO)

try:
    if sys.platform == 'darwin':
        # for Mac M1 version
        DEVICE = torch.device(
            'mps' if not torch.backends.mps.is_available() else 'cpu')
    else:
        # for GPU
        DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
except AttributeError:
    DEVICE = torch.device('cpu')
    

def dump_json(
    obj,
    path: Union[str, Path],
    indent: int = 4,
    mode: str = 'w'
) -> None:
    if not isinstance(path, Path):
        path = Path(path)
    path.parents[0].mkdir(parents=True, exist_ok=True)
    
    with open(path, mode) as io:
        json.dump(obj, io, indent=indent)


def train_dml(args: argparse.Namespace, repeat_ind: int = 0) -> DML:
    pprint(vars(args))
    
    train_loader, val_loader, num_classes = preprocess_data(
        args.data_name, args.batch_size, args.batch_size)

    net_names = []
    for net_name_num in args.network_names.strip().split('-'):
        name, num = net_name_num.split(':')
        num = int(num)
        net_names += [name] * num
        
    networks = [
        get_network(
            net_name, 
            num_classes=num_classes, 
            pretrained=args.network_pretrained).to(DEVICE) 
        for net_name in net_names
    ]
    
    if torch.cuda.device_count() > 1:
        for k in range(len(networks)):
            networks[k] = torch.nn.DataParallel(networks[k])

    optimizers = [
        optim.SGD(
            net.parameters(),
            lr=args.lr,
            nesterov=args.nesterov,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        for _, net in enumerate(networks)
    ]
    
#     schedulers = [
#         optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=args.lr_step_size,
#             gamma=args.lr_decay,
#             last_epoch=-1)
#         for _, optimizer in enumerate(optimizers)
#     ]
    
    schedulers = [
        optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[16, 22],
            gamma=args.lr_decay,
            last_epoch=-1)
        for _, optimizer in enumerate(optimizers)
    ]

    dml = DML(
        networks,
        optimizers,
        schedulers,
        renyi_annealing=args.renyi_annealing,
        renyi_alpha=args.renyi_alpha,
        renyi_alpha_trainable=args.renyi_alpha_trainable)

    if args.validate:
        dml.train(
            train_loader, 
            val_loader, 
            epochs=args.epochs, 
            grad_clip=args.grad_clip)
    else:
        dml.train(
            train_loader, 
            None, 
            epochs=args.epochs,
            grad_clip=args.grad_clip)
        
    if args.validate:
        dump_json(
            dml.history,
            Path(args.history_path) /
                f'{args.data_name}-{args.network_names}-'
                f'{args.renyi_alpha}-history-{repeat_ind}.json')
        
    f1 = F1Score().to(DEVICE)
    acc = Accuracy().to(DEVICE)
    res_dfs = list()
    for k in range(len(networks)):
        pred, labels = dml.infer(
            val_loader, return_logits=True, return_labels=True, net_ind=k)
        pred = torch.argmax(pred, dim=1)
        
        if isinstance(dml.networks[k], torch.nn.DataParallel):
            model_name = type(dml.networks[k].module).__name__
        else:
            model_name = type(dml.networks[k]).__name__
            
        val_acc = np.array(dml.history['val_acc'])[-5:, k] 
        res_dfs.append(
            pd.DataFrame({
                'accuracy': val_acc.mean(),
                'run_ind': repeat_ind,
                'model': f'{model_name}-{k}'
            }, index=[0]))
        
    return dml, res_dfs

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--history_path', type=str, default='./history/')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_step_size', type=int, default=60)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--network_names', type=str,
        help='The network and the corresponding number in the format:'
             ' network1:n_network1-network12:n_network2...')
    parser.add_argument('--renyi_alpha', type=float, default=2)
    parser.add_argument('--renyi_annealing', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--res_path', type=str, default='./results/')
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--grad_clip', action='store_true')                # default False
    parser.add_argument('--nesterov', action='store_true')                 # default False
    parser.add_argument('--network_pretrained', action='store_true')       # default False
    parser.add_argument('--renyi_alpha_trainable', action='store_true')    # default False
    parser.add_argument('--validate', action='store_true')                 # default False


    args = parser.parse_args()
    
    res_dfs = list()
    for i in range(args.repeat):
        dml, res_dfs_ = train_dml(args, repeat_ind=i)
        res_dfs += res_dfs_
        
    res_df = pd.concat(res_dfs, ignore_index=True)
    res_path = (Path(args.res_path) /
        f'{args.data_name}-{args.network_names}-'
        f'{args.renyi_alpha}-result.csv')        

    res_df.to_csv(res_path)

