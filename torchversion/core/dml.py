import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy
from pathlib import Path
from torchmetrics import F1Score, Accuracy
from torch.autograd import Variable
from tqdm import tqdm
from traceback import print_exc
from typing import Union


def cosine_annealing(a_min, a_max, t, t_max):
    return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(t / t_max * math.pi))


class RenyiLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 2,
        eps: float = 1e-8,
        annealing_method: str = None,
        reduction: str = 'batchmean',
        alpha_trainable: bool = False
    ) -> None:
        super(RenyiLoss, self).__init__()
        
        self.alpha = max(0, alpha)
        if alpha_trainable:
            self.alpha = nn.Parameter(
                torch.tensor(self.alpha), requires_grad=True)
            
        self.alpha_max = self.alpha
        self.annealing_method = annealing_method
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits_p: torch.tensor,
        logits_q: torch.tensor
    ) -> Union[float, torch.tensor]:
        """ Renyi loss D(logits_p || logits_q)
        """
        if self.alpha == 0:
            return 0.
      
        p = torch.clamp(F.softmax(logits_p, dim=1), self.eps, 1)
        q = torch.clamp(F.softmax(logits_q, dim=1), self.eps, 1)
      
        if abs(self.alpha - 1) <= 1e-6:
            loss = torch.sum(
                p * (torch.log(p) - torch.log(q)), dim=1)
        else:
            loss = torch.log(
                torch.sum(q * torch.pow(p / q, self.alpha), dim=1)
            ) / (self.alpha - 1)

        if self.reduction == 'batchmean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise(
                'Only `batchmean` and `sum`'
                ' are the supported options for redcution.')

        return loss

    def anneal_step(self, t, t_max):
        if self.annealing_method == 'cos':
            self.alpha = cosine_annealing(0, self.alpha_max, t, t_max)
        print(self.alpha)
            
class DeepMutualLearner(object):
    ''' Deep mutual learning '''
    def __init__(self, networks, optimizers, schedulers=None, **kwargs):
        assert len(networks) == len(optimizers)
        if schedulers is not None:
            assert len(networks) == len(schedulers)

        self.networks = networks
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.n_networks = len(networks)
        self.ckpt_path = Path(kwargs.get('ckpt_path', './ckpt'))
        self.history = {
            'val_loss': list(),
            'val_f1': list(),
            'val_acc': list()}

        device = next(self.networks[0].parameters()).device

        self.dm_loss = RenyiLoss(
            annealing_method=kwargs.get('renyi_annealing', None),
            alpha=kwargs.get('renyi_alpha', 2.),
            alpha_trainable=kwargs.get('renyi_alpha_trainable', False)
        ).to(device)
        self.base_loss = nn.CrossEntropyLoss().to(device)

        self.f1 = F1Score().to(device)
        self.acc = Accuracy().to(device)

    def train(self, train_loader, val_loader=None, epochs=20, **kwargs) -> None:
        ckpt_period = max(epochs // 2, 1)
        
        for epoch in range(epochs):
            logging.info(f'epochs: {epoch+1}/{epochs}')

            self.train_one_epoch(
                train_loader, 
                epoch, 
                epochs, 
                kwargs.get('grad_clip', False))
            self.dm_loss.anneal_step(epoch, epochs)
            
            if self.schedulers is not None:
                for sch in iter(self.schedulers):
                    sch.step()

            if val_loader is not None:
                self.evaluate(val_loader)

            if (epoch > 0 and epoch % ckpt_period == 0) or epoch == epochs - 1:
                self.save_all_models(epoch)

    def train_one_epoch(self, train_loader, epoch, max_epochs, grad_clip=False):
        """ Train the model for 1 epoch of the training set.
            An epoch corresponds to one full pass through the entire
            training set in successive mini-batches.
            This is used by train() and should not be called manually.
        """
        losses, accs = list(), list()

        device = next(self.networks[0].parameters()).device

        for net in iter(self.networks):
            net.train()

        if train_loader.drop_last:
            n_loops = len(train_loader.dataset) // train_loader.batch_size
        else:
            n_loops = math.ceil(
                len(train_loader.dataset) / train_loader.batch_size)
        
        pbar = tqdm(iter(train_loader), total=n_loops)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            #forward pass
            outputs = list()
            for net in iter(self.networks):
                outputs.append(net(inputs))

            for i in range(self.n_networks):
                base_loss = self.base_loss(outputs[i], labels)

                dm_div = 0
                for j in range(self.n_networks):
                    if i == j: continue
                    dm_div += self.dm_loss(Variable(outputs[j]), outputs[i])

                avg_fac = 1 / (self.n_networks-1) if self.n_networks > 1 else 0
                loss = base_loss + avg_fac * dm_div

                # compute gradients and update the parameters
                self.optimizers[i].zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.networks[i].parameters(), max_norm=5)

                self.optimizers[i].step()

                # measure accuracy and record loss
                losses.append(loss.item())
                acc = self.acc(F.softmax(outputs[i], dim=-1), labels)
                accs.append(acc.item())

            # measure elapsed time
            pbar.set_description(
                f'loss: {torch.tensor(losses).mean().item():.4f} -- '
                f'acc: {torch.tensor(accs).mean().item():.4f}')

        return losses

    @torch.no_grad()
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> list:
        scores = list()
        if val_loader is None:
            return scores

        device = next(self.networks[0].parameters()).device

        val_loss, val_f1, val_acc = list(), list(), list()
        for i, net in enumerate(self.networks):
            pred, labels = self.infer(
                val_loader, return_logits=True, return_labels=True, net_ind=i)
            
            loss = self.base_loss(pred, labels).item()

            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            # f1_score = self.f1(pred, labels).item()
            f1_score = 0
            acc_score = self.acc(pred, labels).item()

            val_loss.append(loss)
            val_f1.append(f1_score)
            val_acc.append(acc_score)

            logging.info(
                f'model {i}...val_pure_loss: {loss:.4f},'
                f' val_f1: {f1_score:.4f},'
                f' val_acc: {acc_score:.4f}')

        self.history['val_loss'].append(val_loss)
        self.history['val_f1'].append(val_f1)
        self.history['val_acc'].append(val_acc)

        return copy(val_acc)

    def save_model(self, net_ind: int, epoch: int) -> None:
        net = self.networks[net_ind]
        
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        
        dl_type = 'dml' if self.n_networks > 1 else 'ind'
        path = (
            self.ckpt_path /
            f'{dl_type}_net_{net_ind}_'
            f'{type(net).__name__.lower()}_epoch_{epoch}.pt')
        
        try:
            torch.save(net.state_dict(), path)
        except:
            print_exc()
            
    def save_all_models(self, epoch: int) -> None:
        for k in range(self.n_networks):
            self.save_model(net_ind=k, epoch=epoch)

    def dm_coef(
        self, epoch: int,
        max_epochs: int,
        scale: float = 1.1
    ) -> float:
        return 1

    @torch.no_grad()
    def infer(self, test_loader, return_logits=False, return_labels=False, net_ind=0):
        ''' It returns a tuple where the second element will be empty if
            return_labels is set False
        '''
        net = self.networks[net_ind]
        net.eval()

        device = next(net.parameters()).device

        pred_stack, label_stack = list(), list()
        for inputs, labels in iter(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred_stack.append(net(inputs))
            if return_labels:
                label_stack.append(labels)
        
        pred = torch.vstack(pred_stack)
        if return_labels:
            labels = torch.cat(label_stack)

        if not return_logits:
            pred = F.softmax(pred, dim=1)
        return pred, labels

DML = DeepMutualLearner
