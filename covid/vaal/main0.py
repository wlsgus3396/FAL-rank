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


import pandas as pd ############################################################################################



###############################################################################################################################################
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    
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


    dataset_path = './COVID_dataset/' # <-- Need to be modified properly####################################################
    site_name = ['mgh_e', 'mgh_d', 'mgh_c', 'mgh_b', 'mgh_a', 'korean']##############################################################
    num_site = len(site_name)
    
   
    untrain_dataset=[] 
    t_dataset=[]
    
    unlabeled_indices=[]
    labeled_indices=[]
    df_train=[]
    t_df_train=[]
    for s in range(num_site):
        s_name = site_name[s]
        data_root_path = os.path.join(dataset_path, s_name)
        csv_path = os.path.join(data_root_path, 'ehr_normalized.csv')
        
        
                    
        site_csv = pd.read_csv(csv_path)
        
        df_train.append(site_csv[~site_csv.Fold.isin([0])].reset_index(drop=True))
        t_df_train.append(site_csv[site_csv.Fold.isin([0])].reset_index(drop=True))
        
        
        untrain_dataset.append(plain_COVIDDataSet_binary_balance(data_root = data_root_path, data_list = df_train[s],
                                                            aug = True, aug_params=args.aug_params, size=args.img_size))
        t_dataset.append(plain_COVIDDataSet_binary_balance(data_root = data_root_path, data_list = t_df_train[s],
                                                            aug = True, aug_params=args.aug_params, size=args.img_size))
        unlabeled_indices.append(list(set(np.arange(len(untrain_dataset[s])))))
        labeled_indices.append(list(random.sample(list(unlabeled_indices[s]),len(untrain_dataset[s]//5))))              ## //5
        unlabeled_indices[s]=list(np.setdiff1d(list(unlabeled_indices[s]), labeled_indices[s]))
    
    
    
    total_df_train=pd.concat([df_train[0],df_train[1],df_train[2],df_train[3],df_train[4],df_train[5]],ignore_index=True,axis=0,sort=False)
    total_t_df_train=pd.concat([t_df_train[0],t_df_train[1],t_df_train[2],t_df_train[3],t_df_train[4],t_df_train[5]],ignore_index=True,axis=0,sort=False)
    
    data_root_path = os.path.join(dataset_path, 'total')
    
    total_dataset=plain_COVIDDataSet_binary_balance(data_root = data_root_path, data_list = total_df_train,
                                                            aug = True, aug_params=args.aug_params, size=args.img_size)
    total_t_dataset=COVIDDataSet_binary_balance(data_root = data_root_path, data_list = total_t_df_train,
                                                            aug = False, aug_params=args.aug_params, size=args.img_size)
    
    all_indices=list(set(np.arange(len(total_dataset))))
    t_unlabeled_indices=all_indices
    for k in range(args.num_clients):
        t_unlabeled_indices=np.setdiff1d(list(t_unlabeled_indices), labeled_indices[k])
    
    #test_indices = random.sample(all_indices, len(un_total_dataset)//5)
    test_indices=list(set(np.arange(len(total_t_dataset))))
    #test_indices=list(set(np.arange(len(untrain_dataset[0]))))
    test_sampler = data.sampler.SubsetRandomSampler(test_indices)
    test_dataloader = data.DataLoader(total_t_dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)
    #test_dataloader = data.DataLoader(untrain_dataset[0], sampler=test_sampler, batch_size=args.batch_size, drop_last=False,worker_init_fn=seed_worker)
   
    accuracy=[]
    Solo_accuracy=[]
    accuracies = []
    
    for s in range(num_site):
        accuracies.append(0)

    # dataset with labels available
    
           
    args.cuda=torch.cuda.is_available()




    ########################################################################################################################################################################################3









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
        vae=[]
        discriminator=[]
        stop=[]
        
        unlabeled_dataloader=[]
        querry_dataloader=[] 
        val_dataloader=[]   
            
            
        Itask_model=resnet.resnet18()
        num_ftrs = Itask_model.linear.in_features
        Itask_model.linear=nn.Linear(num_ftrs,args.num_classes)
        Ivae=model.VAE(args.latent_dim)
        Idiscriminator=model.Discriminator(args.latent_dim)

        

        for k in range(args.num_clients):
                
            solver.append(Solver(args, test_dataloader))
            task_model.append(resnet.resnet18())
            task_model[k].linear=nn.Linear(num_ftrs,args.num_classes)
            task_model[k].load_state_dict(Itask_model.state_dict())
            stop.append(1)
        for k in range(args.num_clients):    
            vae.append(model.VAE(args.latent_dim))
            vae[k].load_state_dict(Ivae.state_dict())
            discriminator.append(model.Discriminator(args.latent_dim))
            discriminator[k].load_state_dict(Idiscriminator.state_dict())
        
        R_task_model=copy.deepcopy(task_model)#####################################
        R_vae=copy.deepcopy(vae)###################################################
        R_discriminator=copy.deepcopy(discriminator)####################################################
        lr=args.lr
        random_seed=200+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        AVG_accuracy=0###################################################################################
        n_avg=[]#######################################################################
        for iter in range(args.global_iteration2): 
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            lr*=args.lr_decay
            # need to retrain all the models on the new images
            # re initialize and retrain the models

             
            w_task_model=[]
            w_vae=[]
            w_discriminator=[]
            
            for k in range(args.num_clients):
                     
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))

                
                 ##########################################################################################
                if iter==0:
                    unlabeled_indices[k].sort()
                    labeled_indices[k].sort()
                
                
                    s_name = site_name[k]
                    data_root_path = os.path.join(dataset_path, s_name)
                    csv_path = os.path.join(data_root_path, 'ehr_normalized.csv')   
                    site_csv = pd.read_csv(csv_path)
                    
                    df_train=site_csv[~site_csv.Fold.isin([0])].reset_index(drop=True)
                    
                    df_train=df_train.drop(df_train.index[unlabeled_indices[k]])
                    train_dataset=COVIDDataSet_binary_balance(data_root = data_root_path, data_list = df_train,
                                                                        aug = True, aug_params=args.aug_params, size=args.img_size)
                    sampler = data.sampler.SubsetRandomSampler(list(set(np.arange(len(train_dataset)))))
                    querry_dataloader.append(data.DataLoader(train_dataset,sampler=sampler, batch_size=args.batch_size, drop_last=False,num_workers=args.num_worker,worker_init_fn=seed_worker))
                    
                    
                    df_train=site_csv[~site_csv.Fold.isin([0])].reset_index(drop=True)
                    
                    df_train=df_train.drop(df_train.index[labeled_indices[k]])
                    train_dataset=plain_COVIDDataSet_binary_balance(data_root = data_root_path, data_list = df_train,
                                                                        aug = True, aug_params=args.aug_params, size=args.img_size)
                    sampler = data.sampler.SubsetRandomSampler(list(set(np.arange(len(train_dataset)))))
                    unlabeled_dataloader.append(data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,num_workers=args.num_worker,worker_init_fn=seed_worker))
                
                    n_avg.append(len(labeled_indices[k]))
                    val_dataloader.append(data.DataLoader(train_dataset,sampler=sampler, batch_size=args.batch_size, drop_last=False)) ###################################
                ################################################################################################
                

                _,task_model[k],vae[k],discriminator[k],stop[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator[k],unlabeled_dataloader[k],lr,0,iter)
                w_task_model.append(task_model[k].state_dict())
                w_vae.append(vae[k].state_dict())
                w_discriminator.append(discriminator[k].state_dict())
                
            
            global_task_model=copy.deepcopy(task_model[0])
            global_task_model.load_state_dict(FedAvg(w_task_model,n_avg))
            global_vae=copy.deepcopy(vae[0])
            global_vae.load_state_dict(FedAvg(w_vae,n_avg))
            global_discriminator=copy.deepcopy(discriminator[0])
            global_discriminator.load_state_dict(FedAvg(w_discriminator,n_avg))
            
            if iter!=args.global_iteration2-1:
                for k in range(args.num_clients):
                    task_model[k]=copy.deepcopy(global_task_model)
                    vae[k]=global_vae
                    discriminator[k]=global_discriminator
                    
            task_model[0] = task_model[0].cuda()                
            final_accuracy = solver[0].test(task_model[0])
            
                        
            if final_accuracy > AVG_accuracy:############################################3
                AVG_accuracy=final_accuracy ######################################################################################
                for k in range(args.num_clients):####################################
                    R_task_model[k]=task_model[k]#################################################
                    R_vae[k]=vae[k]##############################################################
                    R_discriminator[k]=discriminator[k]    #####################################################################
           
            
            
            
            if sum(stop)==0:
                break
        
       
        accuracy.append(AVG_accuracy)
                
                
        
        
        
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
        vae=[]
        discriminator=[]
        stop=[]
        
        Itask_model=resnet.resnet18()
        num_ftrs = Itask_model.linear.in_features
        Itask_model.linear=nn.Linear(num_ftrs,args.num_classes)
        Ivae=model.VAE(args.latent_dim)
        Idiscriminator=model.Discriminator(args.latent_dim)

        

        for k in range(args.num_clients):
                
            solver.append(Solver(args, test_dataloader))
            task_model.append(resnet.resnet18())
            task_model[k].linear=nn.Linear(num_ftrs,args.num_classes)
            task_model[k].load_state_dict(Itask_model.state_dict())
            stop.append(1)
        for k in range(args.num_clients):    
            vae.append(model.VAE(args.latent_dim))
            vae[k].load_state_dict(Ivae.state_dict())
            discriminator.append(model.Discriminator(args.latent_dim))
            discriminator[k].load_state_dict(Idiscriminator.state_dict())
        
        R_task_model=copy.deepcopy(task_model)#####################################
        R_vae=copy.deepcopy(vae)###################################################
        R_discriminator=copy.deepcopy(discriminator)####################################################
        lr=args.lr
        random_seed=200+1000*args.K+iiter
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        AVG_accuracy=0###################################################################################
        for iter in range(args.global_iteration2): 
            # need to retrain all the models on the new images
            # re initialize and retrain the models
            lr*=args.lr_decay
            # need to retrain all the models on the new images
            # re initialize and retrain the models

             
            w_task_model=[]
            w_vae=[]
            w_discriminator=[]
            
            for k in range(args.num_clients):
                     
                print('Current global iteration1: {}'.format(iiter+1))
                print('Current global iteration2: {}'.format(iter+1))
                print('Client: {}'.format(k+1))

                

                _,task_model[k],vae[k],discriminator[k],stop[k] = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator[k],unlabeled_dataloader[k],lr,1,iter)
                w_task_model.append(task_model[k].state_dict())
                w_vae.append(vae[k].state_dict())
                w_discriminator.append(discriminator[k].state_dict())
                
            
            global_task_model=copy.deepcopy(task_model[0])
            global_task_model.load_state_dict(FedAvg(w_task_model,n_avg))
            global_vae=copy.deepcopy(vae[0])
            global_vae.load_state_dict(FedAvg(w_vae,n_avg))
            global_discriminator=copy.deepcopy(discriminator[0])
            global_discriminator.load_state_dict(FedAvg(w_discriminator,n_avg))
            
            if iter!=args.global_iteration2-1:
                for k in range(args.num_clients):
                    task_model[k]=copy.deepcopy(global_task_model)
                    vae[k]=global_vae
                    discriminator[k]=global_discriminator
                    
            task_model[0] = task_model[0].cuda()                
            final_accuracy = solver[0].test(task_model[0])
            
                        
            if final_accuracy > AVG_accuracy:############################################3
                AVG_accuracy=final_accuracy ######################################################################################
                for k in range(args.num_clients):####################################
                    R_task_model[k]=task_model[k]#################################################
                    R_vae[k]=vae[k]##############################################################
                    R_discriminator[k]=discriminator[k]    #####################################################################
           
            
            
            
            if sum(stop)==0:
                break
            
            
            
            
            
            

        if iiter!=args.global_iteration1-1:
            for k in range(args.num_clients):
                sampled_indices = solver[k].sample_for_labeling(R_task_model[k],R_vae[k], R_discriminator[k] ,unlabeled_dataloader[k])
                labeled_indices[k] = list(set().union(list(labeled_indices[k]) ,list(sampled_indices)))
                unlabeled_indices[k]=list(np.setdiff1d(list(unlabeled_indices[k]), list(sampled_indices)))
        

        solver=[]
        task_model=[]
        vae=[]
        discriminator=[]
        
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
            vae.append(model.VAE(args.latent_dim))
            
            discriminator.append(model.Discriminator(args.latent_dim))
            
            
        
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
                
                Soloaccuracy, task_model[k],_,_ = solver[k].train(querry_dataloader[k],val_dataloader[k],task_model[k],vae[k],discriminator[k],unlabeled_dataloader[k],lr,0,iter)
                Solo_accuracy[iiter]+=Soloaccuracy
                
        Solo_accuracy[iiter]=Solo_accuracy[iiter]/args.num_clients


    
        print('Final Fed accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, AVG_accuracy))
        print('Final Solo accuracy at the {}-th global iteration of data is: {:.2f}'.format(iiter+1, Solo_accuracy[iiter]))
        


    #torch.save(accuracy, os.path.join(args.out_path, args.log_name))
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

