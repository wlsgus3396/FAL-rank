import torch
from torchvision import datasets, transforms, models
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import torch.nn as nn

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import resnet
from solver import Solver
from solver_2 import Solver2
from utils import *
import arguments
import copy
from FedAVG import FedAvg




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cifar_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
def mnist_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.286), (0.353))
       ])
def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)

        train_dataset = CIFAR10(args.data_path)
        untrain_dataset=plain_CIFAR10(args.data_path)
        args.num_images = 50000
        args.num_val = 5000

        args.num_classes = 10
        
    elif args.dataset == 'mnist':
        test_dataloader = data.DataLoader(
                datasets.FashionMNIST(args.data_path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)
        untrain_dataset=plain_MNIST(args.data_path)
        train_dataset = MNIST(args.data_path)
        
        
        args.num_images = 50000
        args.num_val = 5000

        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)

        train_dataset = CIFAR100(args.data_path)
        untrain_dataset=plain_CIFAR100(args.data_path)
        args.num_images = 50000
        args.num_val = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_val = 128120
        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError


    GPU_NUM = args.gpu # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    random_seed=100+1000*args.K
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





    all_indices = set(np.arange(args.num_images)) 
    val_indices = random.sample(all_indices, args.num_val) #validation
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)
    
    #all_indices = np.setdiff1d(list(all_indices), val_indices) #exclude
    labeled_indices=[]
    unlabeled_indices=[]
    remaining_indices=[]





    accuracy=[]
    Solo_accuracy=[]
    accuracies = []
    
    #Itask_model=vgg.vgg16_bn(num_classes=args.num_classes)

    all_indices1=copy.deepcopy(all_indices)
    for k in range(args.num_clients):
        
        unlabeled_indices.append(list(random.sample(list(all_indices1), args.unlabeledbudget)))
        all_indices1=np.setdiff1d(list(all_indices1), unlabeled_indices[k])
        labeled_indices.append(list(random.sample(list(unlabeled_indices[k]), args.initial_budget))) #initial budget
        unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), labeled_indices[k]))
        accuracies.append(0)

    # dataset with labels available
    
    
    val_dataloader = data.DataLoader(train_dataset, 
            batch_size=args.batch_size, drop_last=True)
            

    args.cuda=torch.cuda.is_available()





    for iiter in range(args.global_iteration1): 
        random_seed=100+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        Solo_accuracy.append(0)
        solver=[]
        task_model=[]
        FC1=[]
        FC2=[]
        stop=[]
        unlabeled_dataloader=[]
        querry_dataloader=[] 
        val_dataloader=[]

            
            
        Itask_model=resnet.resnet18()
        num_ftrs = Itask_model.linear.in_features
        Itask_model.linear=nn.Linear(num_ftrs,args.num_classes)

        
 



        for k in range(args.num_clients):
                
            solver.append(Solver(args, test_dataloader))
                
            task_model.append(resnet.resnet18())
            task_model[k].linear=nn.Linear(num_ftrs,args.num_classes)
            task_model[k].load_state_dict(Itask_model.state_dict())
            stop.append(1)
        for k in range(args.num_clients):
            FC1.append(nn.Linear(num_ftrs,args.num_classes))
            FC1[k].load_state_dict(Itask_model.linear.state_dict())
            FC2.append(nn.Linear(num_ftrs,args.num_classes))
            FC2[k].load_state_dict(Itask_model.linear.state_dict())
            
        
        
        
        lr=args.lr
        random_seed=200+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        AVG_accuracy=0################################################################################################
        n_avg=[]#######################################################################
        for iter in range(args.global_iteration2): 
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            lr*=args.lr_decay
            # need to retrain all the models on the new images
            # re initialize and retrain the models

                
            w_task_model=[]

            for k in range(args.num_clients):     
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))

        
                if iter==0:
                    unlabeled_indices[k].sort()
                    labeled_indices[k].sort()
                    unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices[k])
                    unlabeled_dataloader.append(data.DataLoader(untrain_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker))  
                    sampler = data.sampler.SubsetRandomSampler(labeled_indices[k])
                    querry_dataloader.append(data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker))
                    n_avg.append(len(labeled_indices[k]))#################################################
                    val_dataloader.append(data.DataLoader(untrain_dataset, sampler=sampler,batch_size=args.batch_size, drop_last=False))
                

                
                _, task_model[k],_,_,stop[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],FC1[k],FC2[k],unlabeled_dataloader[k],lr,0,iter)
                


                w_task_model.append(task_model[k].state_dict())
            
            
            global_task_model=copy.deepcopy(task_model[0])
            global_task_model.load_state_dict(FedAvg(w_task_model,n_avg))
            
            
            
            if iter!=args.global_iteration2-1:
                for k in range(args.num_clients):
                    task_model[k]=copy.deepcopy(global_task_model)
                     
            
            
            
            task_model[0] = task_model[0].cuda()                
            final_accuracy = solver[0].test(task_model[0])
            
            
            
            
            if final_accuracy > AVG_accuracy:############################################
                AVG_accuracy=final_accuracy ################################################
            
            
            if sum(stop)==0:
                break
                
         #####################################################################################   
        accuracy.append(AVG_accuracy)

 
        solver=[]
        task_model=[]
        FC1=[]
        FC2=[]
        
        random_seed=300+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for k in range(args.num_clients):
            solver.append(Solver2(args, test_dataloader))
            task_model.append(resnet.resnet18())
            task_model[k].linear=nn.Linear(num_ftrs,args.num_classes)
            
        for k in range(args.num_clients):
            FC1.append(nn.Linear(num_ftrs,args.num_classes))
            FC1[k].load_state_dict(task_model[k].linear.state_dict())
            FC2.append(nn.Linear(num_ftrs,args.num_classes))
            FC2[k].load_state_dict(task_model[k].linear.state_dict())
            
            
        lr=args.lr_solo

        random_seed=400+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for iter in range(1): 
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            lr*=args.lr_decay
            # need to retrain all the models on the new images
            # re initialize and retrain the models
                
                
                
            for k in range(args.num_clients):  
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))
                print('Solo learning')     
                Soloaccuracy, task_model[k],FC1[k],FC2[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],FC1[k],FC2[k],unlabeled_dataloader[k],lr,0,iter)
                Solo_accuracy[iiter]+=Soloaccuracy
                    

            

        Solo_accuracy[iiter]=Solo_accuracy[iiter]/args.num_clients   


        
        DL_dis=0
        Num_dis=0
        DL_item=[]
        
        for k in range(args.num_clients):
            DL_dis+=solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])
            Num_dis+=len(labeled_indices[k])
            DL_item.append(solver[k].DLcal(task_model[k],FC1[k], FC2[k],querry_dataloader[k])/len(labeled_indices[k])/args.num_classes)
        
        DL_dis=DL_dis/Num_dis/args.num_classes
                            
        if iiter!=args.global_iteration1-1:
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(task_model[k],FC1[k], FC2[k], DL_dis ,DL_item[k] ,unlabeled_dataloader[k])
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))


        print('Final Fed accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, AVG_accuracy))
        print('Final Solo accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, Solo_accuracy[iiter]))

    accuracy = np.array(accuracy)
    A ="\n".join(map(str, accuracy))
    f = open('./results/{}_numclients_{}_lr_{}_lr_decay_{}_Fedacc_{}.csv'.format(args.execute,args.num_clients,args.lr,args.lr_decay,args.K),'w')
    f.write(A)
    f.close()

    Solo_accuracy=np.array(Solo_accuracy)
    A ="\n".join(map(str, Solo_accuracy))
    f = open('./results/{}_numclients_{}_lr_{}_lr_decay_{}_Soloacc_{}.csv'.format(args.execute,args.num_clients,args.lr,args.lr_decay,args.K),'w')
    f.write(A)
    f.close()
if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

