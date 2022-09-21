import logging
import numpy as np
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy 
from pathlib import Path
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy
from torch.autograd import Variable
from traceback import print_exc
from typing import Union

from utils import * 

#torch.autograd.set_detect_anomaly(True)

class RenyiLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 2,
        eps: float = 1e-8,
        reduction: str = 'batchmean',
        alpha_trainable: bool = False
    ) -> None:
        super(RenyiLoss, self).__init__()

        self.alpha = alpha
        if alpha_trainable:
            self.alpha = nn.Parameter(
                torch.tensor(self.alpha), requires_grad=True)
        #self.renyi_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)          

        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits_p: torch.tensor,
        logits_q: torch.tensor
    ) -> torch.tensor:
        """ Renyi loss D(logits_p || logits_q)
        """
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
        elif self.reduction == 'batchsum':
            loss = loss.sum()
        else:
            raise(
                'Only `batchmean` and `batchsum`'
                ' are the supported options for redcution.')

        return loss


class DeepMutualLearner(object):
    ''' Deep mutual learning '''
    def __init__(self, networks, optimizers=None, schedulers=None, **kwargs):
        #assert len(networks) == len(optimizers)
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

        self.renyi_alpha = kwargs.get('renyi_alpha', 2.)
        renyi_alpha_trainable = kwargs.get('renyi_alpha_trainable', False)
        #print(renyi_alpha_trainable, '---') 
        self.dm_loss = RenyiLoss(
            alpha=self.renyi_alpha,
            alpha_trainable=renyi_alpha_trainable).to(device)
        #self.base_loss = nn.NLLLoss().to(device)
        self.base_loss = nn.CrossEntropyLoss().to(device)

        self.f1 = F1Score().to(device)
        self.acc = Accuracy().to(device)


    def run_epoch(self, train_iterator, val_iterator, test_iterator, epoch, config):
        max_epoch = config.max_epochs
        train_losses = {}
        val_accuracies = []
        test_accuracies = []
        losses = {}

        device = next(self.networks[0].parameters()).device        

        for net in iter(self.networks):
            net.train()

        # Dynamically adjust alpha
        if epoch != 0:
            # Exponential decay/increase, no change if 1.0
            self.dm_loss.alpha *= config.scale
            # Cosine annealing
            if config.ca == 1:
                alpha_min = 0.5
                alpha_max = 2.0
                self.dm_loss.alpha = alpha_min + 0.5*(alpha_max-alpha_min)*(1.0+math.cos((np.float(epoch)/max_epoch)*math.pi))
            # Sine annealing
            #self.dm_loss.alpha = self.renyi_alpha * (1.0 + math.sin((np.float(epoch)/max_epoch) * math.pi))
            #print('epoch', epoch, 'alpha', self.renyi_alpha, 'alpha in Renyi', self.dm_loss.alpha)

        for i, batch in enumerate(train_iterator):
            x = batch.text.to(device) 
            y = (batch.label).to(device=device, dtype=torch.long)

            #if torch.cuda.is_available():
            #    x = batch.text.cuda()
            #    y = (batch.label-1).type(torch.cuda.LongTensor)
            #else:
            #    x = batch.text
            #    y = (batch.label-1).type(torch.LongTensor)

            # forward pass
            outputs = list()
            for net in iter(self.networks):
                outputs.append(net(x))           
 
            for j, model in enumerate(self.networks):
                try:
                    base_loss = self.base_loss(outputs[j], y)
                except Exception as e:
                    print('Error for calculating base loss', e)
                    #print(x)
                    #print(y)
                # Problem occurs
                dm_div = torch.tensor(0., device=device)
                
                for k in range(self.n_networks):
                    if j == k: continue
                    dm_div += self.dm_loss(Variable(outputs[k]), outputs[j])

                dm_coef = self.dm_coef(epoch, model.config.max_epochs)
                avg_fac = (
                    dm_coef/(self.n_networks-1) if self.n_networks > 1 else 0)
                
                #print('base_loss', base_loss.item(), 'dml loss', (avg_fac*dm_div).item())
                loss = base_loss + avg_fac * dm_div
                #loss = base_loss
                
                self.optimizers[j].zero_grad()
                #self.dm_loss.renyi_optimizer.zero_grad()
                loss.backward()
                #self.dm_loss.renyi_optimizer.step()
                self.optimizers[j].step()

                #self.dm_loss.renyi_optimizer.zero_grad()


                if j in losses: 
                    losses[j].append(loss.data.cpu().numpy())
                else:
                    losses[j] = [loss.data.cpu().numpy()]
                #model.optimizer.step()
                
                if i % 100 == 0:
                    #print('Iter: {}'.format(i+1))
                    avg_train_loss = np.mean(losses[j])
                    if j in train_losses:
                        train_losses[j].append(avg_train_loss)
                    else:
                        train_losses[j] = [avg_train_loss]
                    #print("\t{}-th Model Average training loss: {:.5f}".format(j+1, avg_train_loss))
                    #pbar.set_description(f'Iter: {i+1} -- {j+1}-th model average training loss: {avg_train_loss:.4f}')
                    losses[j] = []
                
                    print(f"Iter: {i+1} -- Base loss: {base_loss.item()} -- DML loss: {(avg_fac*dm_div).item()}")
                    
            
                    # Evalute Accuracy on validation set
                    if config.eval_val_every100 == 1:
                        val_accuracy = evaluate_model(model, val_iterator)
                        print(f"Validation Accuracy: {val_accuracy}")
                        model.train()
                    #print('base_loss', base_loss.item(), 'dml loss', (avg_fac*dm_div).item())
        
        # Validate for the epoch
        for i, model in enumerate(self.networks):
            val_accuracy = evaluate_model(model, val_iterator)
            val_accuracies.append(val_accuracy)
            test_accuracy = evaluate_model(model, test_iterator)
            test_accuracies.append(test_accuracy)
            model.train()

        #print(self.dm_loss.alpha.item())        
        return train_losses, val_accuracies, test_accuracies      
        

    def train(self, train_loader, val_loader=None, epochs=20, **kwargs) -> None:
        ckpt_period = max(epochs // 2, 1)
        
        for epoch in range(epochs):
            logging.info(f'epochs: {epoch+1}/{epochs}')

            self.train_one_epoch(train_loader, epoch, epochs)   
            if self.schedulers is not None:
                for sch in self.schedulers:
                    sch.step()

            if val_loader is not None:
                self.evaluate(val_loader)

            if (epoch > 0 and epoch % ckpt_period == 0) or epoch == epochs - 1:
                self.save_all_models(epoch)


    def train_one_epoch(self, train_loader, epoch, max_epochs):
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
            n_loops = int(
                math.ceil(len(train_loader.dataset) / train_loader.batch_size))
        
        pbar = tqdm(train_loader, total=n_loops)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device=device), labels.to(device=device)

            #forward pass
            outputs = list()
            for net in iter(self.networks):
                outputs.append(net(inputs))

            for i in range(self.n_networks):
                base_loss = self.base_loss(outputs[i], labels)

                dm_div = torch.tensor(0., device=device)
                for j in range(self.n_networks):
                    if i == j: continue
                    dm_div += self.dm_loss(Variable(outputs[j]), outputs[i])

                dm_coef = self.dm_coef(epoch, max_epochs)
                avg_fac = (
                    dm_coef/(self.n_networks-1) if self.n_networks > 1 else 0)

                loss = base_loss + avg_fac * dm_div
                print('loss:', loss.item(), 'baseloss:', base_loss.item())

                # compute gradients and update SGD
                self.optimizers[i].zero_grad()
                loss.backward()
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
            net.eval()

            pred_stack, label_stack = list(), list()
            for _, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                pred_stack.append(net(inputs))
                label_stack.append(labels)

            pred = torch.vstack(pred_stack)
            labels = torch.concat(label_stack)
            
            loss = self.base_loss(pred, labels).item()

            pred = F.softmax(pred, dim=1)
            f1_score = self.f1(pred, labels).item()
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
            (f'{dl_type}_net_{net_ind}'
             f'{type(net).__name__.lower()}_epoch_{epoch}.pt'))
        
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
        scale: float = 1.1, 
        decay: float = 2.
    ) -> float:
        return 1

    @torch.no_grad()
    def infer(self, test_loader, return_labels=False, net_ind=0):
        ''' It returns a tuple where the second element will be empty if 
            return_labels is set False
        '''
        net = self.networks[net_ind] 
        net.eval()

        device = next(net.parameters()).device

        pred_stack, label_stack = list(), list()
        for _, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred_stack.append(net(inputs))
            if return_labels:
                label_stack.append(labels)
        
        pred = torch.vstack(pred_stack)
        if return_labels:
            labels = torch.cat(label_stack)

        return F.softmax(pred, dim=1), labels 

DML = DeepMutualLearner
