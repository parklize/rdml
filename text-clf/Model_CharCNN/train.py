import time
import os
import numpy as np
import sys
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

from utils import *
from model import *
from config import Config
from torch import nn
from tqdm import tqdm
from pprint import pprint
from dml import DML

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


if __name__=='__main__':
    config = Config()
    train_file = '../data/ag_news.train'
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    test_file = '../data/ag_news.test'
    if len(sys.argv) > 2:
        test_file = sys.argv[2]
    if len(sys.argv) > 3:
        config.renyi_alpha = float(sys.argv[3])
        config.networks = 2
    if len(sys.argv) > 4:
        config.scale = float(sys.argv[4])
    if len(sys.argv) > 5:
        config.ca = int(sys.argv[5])
    
    print('\n====Configuration====')
    for v in dir(config):
        if '__' not in v:
            print(v, config.__getattribute__(v))
    print(f'\ntraining with {train_file}')
    print(f'testing with {test_file}')

    dataset_name = train_file.replace('../data/','').split('.')[0]
    print(f'\nDataset name: {dataset_name}')

    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)
    
    # Create corresponding folder for results
    fn = ''
    if config.renyi_alpha == 0:
        fn += 'ind'
    else:
        fn += 'renyi'
        fn += (str(config.renyi_alpha) + '-')
        fn += ('scale' + str(config.scale) + '-')
        fn += ('ca' + str(config.ca))
    if not os.path.exists(f'results/{dataset_name}/{fn}'):
        os.makedirs(f'results/{dataset_name}/{fn}')

    # Create Model with specified optimizer and loss function
    ##############################################################
    models = []
    for i in range(config.networks):
        models.append(CharCNN(config, len(dataset.vocab), dataset.embeddings))
        models[-1].to(DEVICE)

    for model_ind, model in enumerate(models):
    	#Print
    	print('\n', model)
    	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    	print(f'\ntotal params:{pytorch_total_params}\n')
    #	
    #	model.train()

    #	optimizer = optim.SGD(model.parameters(), lr=config.lr)
    #	NLLLoss = nn.NLLLoss()
    #	model.add_optimizer(optimizer)
    #	model.add_loss_op(NLLLoss)
    
    optimizers = [
        optim.Adam(
            model.parameters(),
            lr=config.lr)
        for _, model in enumerate(models)
    ]

    schedulers = [
        optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, 
            gamma=config.lr_decay, 
            last_epoch=-1, 
            verbose=True)
            for _, optimizer in enumerate(optimizers)
    ]
    
    dml = DML(
        models,
        optimizers,
        schedulers,
        renyi_alpha=config.renyi_alpha, 
        renyi_alpha_trainable=False) 
        
    ##############################################################
    
    train_losses = [[] for i in range(len(models))]
    val_accuracies_epochs = [[] for i in range(len(models))]
    test_accuracies_epochs = [[] for i in range(len(models))]
    best_val_acc = [0.0] * len(models)
    patience = [0] * len(models)
    
    #pbar = tqdm(range(config.max_epochs), total=config.max_epochs)
    for i in range(config.max_epochs):
        stime = time.time()
        print ("\nEpoch: {}".format(i))
        train_loss,val_accuracies,test_accuracies = dml.run_epoch(
            dataset.train_iterator, 
            dataset.val_iterator,
            dataset.test_iterator, 
            i, 
            config)
        for j in range(len(models)):
            #print(j,'train_losses', train_losses)
            #print(j,'train_loss',  train_loss)
            train_losses[j] += train_loss[j]
            val_accuracies_epochs[j].append(val_accuracies[j])
            test_accuracies_epochs[j].append(test_accuracies[j])
            #np_params = torch.nn.utils.parameters_to_vector([p for p in models[j].parameters() if p.requires_grad]).detach().numpy()
            #print(np_params.shape)
            #print(np_params[:5])
            #np.savetxt(f'results/{dataset_name}/{fn}/params-ep{i}-m{j}.txt', np_params, delimiter='\n')
        # Save the best model on val
        for j, val_acc in enumerate(val_accuracies):
            if val_acc >= best_val_acc[j]:
                print(f'Best accuracy of {j+1}-th model updated {best_val_acc[j]:.4f}=>{val_acc:.4f}')
                best_val_acc[j] = val_acc
                torch.save(models[j].state_dict(), 'fastText-{}.pth'.format(j))
                patience[j] = 0
            else:
                print(f"{j+1}-th model didn't improve from {best_val_acc[j]} -- {val_acc:.4f}")
                patience[j] += 1
        # If both patience >= 5, stop
        #print('current patience', patience)
        pat_max_count = sum([x>=config.patience_th for x in patience])
        if pat_max_count == len(models):
            print(f'\nEarly stopping with all patience >= {config.patience_th}')
            break
 
        if dml.schedulers is not None:
            for sch in dml.schedulers:
                sch.step()
        
        etime = time.time()
        print(f'{etime-stime} seconds passed...')
    
    # Create corresponding folder for results
    #fn = ''
    #if config.renyi_alpha == 0:
    #    fn += 'ind'
    #else:
    #    fn += 'renyi'
    #    fn += (str(config.renyi_alpha) + '-')
    #    fn += ('scale' + str(config.scale) + '-')
    #    fn += ('ca' + str(config.ca))
    #if not os.path.exists(f'results/{dataset_name}/{fn}'):
    #    os.mkdir(f'results/{dataset_name}/{fn}')

    # Plot train loss, val acc
    for i in range(len(models)):
        plt.plot(train_losses[i],'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Training loss')
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}/{fn}/training_loss_avg_100iter-{i}-th.png')
        plt.close()
        with open(f'results/{dataset_name}/{fn}/training_loss_avg_100iter-{i}-th.txt', 'w') as fp:
            fp.write('\n'.join([str(x) for x in train_losses[i]]))
        
        plt.plot(val_accuracies_epochs[i],'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Validation accuracy')
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}/{fn}/val_acc_ep-{i}-th.png')
        plt.close()
        with open(f'results/{dataset_name}/{fn}/val_acc_ep-{i}-th.txt', 'w') as fp:
            fp.write('\n'.join([str(x) for x in val_accuracies_epochs[i]]))

        plt.plot(test_accuracies_epochs[i],'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Test accuracy')
        plt.tight_layout()
        plt.savefig(f'results/{dataset_name}/{fn}/test_acc-ep-{i}-th.png')
        plt.close()
        with open(f'results/{dataset_name}/{fn}/test_acc_ep-{i}-th.txt', 'w') as fp:
            fp.write('\n'.join([str(x) for x in test_accuracies_epochs[i]]))

    for model_ind, model in enumerate(models):
    	train_acc = evaluate_model(model, dataset.train_iterator)
    	val_acc = evaluate_model(model, dataset.val_iterator)
    	test_acc = evaluate_model(model, dataset.test_iterator)
        
    	print ('-----------------------------')
    	print ('{}-th model Performance'.format(model_ind+1))
    	print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    	print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    	print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    
    print('\n-----------------------------')
    print('From loaded best models')
    for i in range(len(models)):
        models[i].load_state_dict(torch.load(f'fastText-{i}.pth'))
        train_acc = evaluate_model(models[i], dataset.train_iterator)
        val_acc = evaluate_model(models[i], dataset.val_iterator)
        test_acc = evaluate_model(models[i], dataset.test_iterator)
        
        print ('-----------------------------')
        print ('{}-th model Performance'.format(i+1))
        print ('Final Training Accuracy: {:.4f}'.format(train_acc))
        print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
        print ('Final Test Accuracy: {:.4f}'.format(test_acc))

        with open(f'results/{dataset_name}/{fn}/best_results-{i+1}.txt', 'w') as f:
            for v in dir(config):
                if '__' not in v:
                    f.write(f'{v}, {config.__getattribute__(v)}\n')
            f.write('\n{}-th model Performance'.format(i+1))
            f.write('\nFinal Training Accuracy: {:.4f}'.format(train_acc))
            f.write('\nFinal Validation Accuracy: {:.4f}'.format(val_acc))
            f.write('\nFinal Test Accuracy: {:.4f}'.format(test_acc))
