import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms


def get_breast_cancer():
    df = pd.read_csv(f'./data/ionosphere/data.csv', sep=',', header=None)
    # enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # val = enc.fit_transform([df.values[:, -1]])
    # print(val)
    # df.iloc[:, -1] = val[0]
    
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == 'b' else 0)
    n_classes = df.iloc[:, -1].unique().shape[0]
    
    train_df, test_df = train_test_split(df, test_size=0.3)
    train_data, train_label = train_df.values[:, :-1], train_df.values[:, -1]
    test_data, test_label = test_df.values[:, :-1], test_df.values[:, -1]
    
    train_loader = data.DataLoader(
        data.TensorDataset(
            torch.tensor(train_data, dtype=torch.float32), 
            torch.tensor(train_label, dtype=torch.long)),
        batch_size=16,
        shuffle=True)
    
    test_loader = data.DataLoader(
        data.TensorDataset(
            torch.tensor(test_data, dtype=torch.float32), 
            torch.tensor(test_label, dtype=torch.long)),
        batch_size=16,
        shuffle=False)
    
    return train_loader, test_loader, train_data.shape[1], n_classes 


def preprocess_data(
    data_name,
    train_batch_size: int,
    test_batch_size: int
):
    num_classes = 100 
    if data_name == 'cifar10':
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])

        test_trans = transforms.Compose([
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10',
            train=True,
            download=True,
            transform=train_trans)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10',
            train=False,
            download=False,
            transform=test_trans)
    elif data_name == 'cifar100':
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])

        test_trans = transforms.Compose([
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data/cifar100',
            train=True,
            download=True,
            transform=train_trans)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data/cifar100',
            train=False,
            download=False,
            transform=test_trans)
    elif data_name == 'flowers102':
        train_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        train_dataset = torchvision.datasets.Flowers102(
            root='./data/flowers102',
            split='train',
            download=True,
            transform=train_trans)
        test_dataset = torchvision.datasets.Flowers102(
            root='./data/flowers102',
            split='test',
            download=False,
            transform=test_trans)
    elif data_name == 'dtd':
        train_trans = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        train_dataset = torchvision.datasets.DTD(
            root='./data/dtd',
            split='train',
            download=True,
            transform=train_trans)
        test_dataset = torchvision.datasets.DTD(
            root='./data/dtd',
            split='test',
            download=False,
            transform=test_trans)
        
        num_classes = 47
    elif data_name == 'caltech256':
        train_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        train_dataset = torchvision.datasets.Caltech256(
            root='./data/caltech256',
            download=True,
            transform=train_trans)
        
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size])
        
        num_classes = 47
    
    elif data_name == 'food101':
        train_trans = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        train_dataset = torchvision.datasets.Food101(
            root='./data/food101',
            split='train',
            download=True,
            transform=train_trans)
        test_dataset = torchvision.datasets.Food101(
            root='./data/food101',
            split='test',
            download=False,
            transform=test_trans)
        
        num_classes = 1000
    else:
        raise(f'{data_name} is not yet supported')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False, 
        pin_memory=True)

    return train_loader, test_loader, num_classes
