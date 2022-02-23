import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score
import vgg
import sampler
import copy
import random 






class Solver2:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sampler = sampler.AdversarySampler(self.args)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    
    def train(self, querry_dataloader, val_dataloader, task_model,FC1,FC2, unlabeled_dataloader,lr_input,mode,iter):
        R_task_model=copy.deepcopy(task_model)#####################################################
        R_FC1=copy.deepcopy(FC1)################################################################
        R_FC2=copy.deepcopy(FC2)#################################################################
        final_accuracy=0##########################################################################################################################################################
        
        self.args.train_iterations = int(self.args.global_iteration2*len(querry_dataloader)* self.args.train_epochs)
        
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)
        
        
        #optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input)
        #optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)
        optim_task_model = optim.SGD(task_model.parameters(), lr=lr_input ,weight_decay=5e-4, momentum=0.9)
    
        task_model.train()
        if self.args.cuda:
            task_model = task_model.cuda()
        
    
        
    
        if mode==0:
            num_ftrs=task_model.linear.in_features
            task_model1=torch.nn.Linear(num_ftrs, self.args.num_classes)
            task_model2=torch.nn.Linear(num_ftrs, self.args.num_classes)
            
            
            task_model1.load_state_dict(FC1.state_dict())
            task_model2.load_state_dict(FC2.state_dict())
            for iter_count in range(self.args.train_iterations):
                
                if iter_count % (len(querry_dataloader)* self.args.train_epochs) == 0:
                    for param in optim_task_model.param_groups:
                        param['lr'] = param['lr'] * self.args.lr_decay

                labeled_imgs, labels = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)
                

                if self.args.cuda:
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

                # task_model step
                preds,_=task_model(labeled_imgs)

                task_loss = self.ce_loss(preds, labels)
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()
                if iter_count % 100 == 0:
                    print('Current training iteration: {}'.format(iter_count))
                    print('Current task model loss: {:.4f}'.format(task_loss.item()))
                
                
                if iter_count % (len(querry_dataloader)* self.args.train_epochs) == 0 and task_loss.item()<0.001:##################################################################
                        if self.args.cuda:
                            task_model = task_model.cuda()
                            if self.test(task_model)>final_accuracy:
                                final_accuracy = max(final_accuracy,self.test(task_model))             
                                R_task_model=task_model #####################################################
                                R_FC1=task_model1 ################################################################
                                R_FC2=task_model2 #################################################################_   
                        if self.validate(task_model, querry_dataloader)<0.0005:
                            break#################################################################################
                
                    
        
            
        
            
            
        
        
        print('acc: ', final_accuracy)
        return final_accuracy,R_task_model,R_FC1, R_FC2############################################################################################
        




    def sample_for_labeling(self, task_model,FC1, FC2,  unlabeled_dataloader,querry_dataloader,data_unlabeled,subset,labeled_set,labeled_data_size):
        querry_indices = self.sampler.sample(task_model,FC1, FC2,
                                             unlabeled_dataloader,querry_dataloader,data_unlabeled,subset,labeled_set,labeled_data_size, 
                                             self.args.cuda,self.args.execute)

        return querry_indices
                





    def validate(self, task_model, loader):
        task_model.eval()
        loss=[]
        for imgs, labels, _ in loader:
            if self.args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                preds,_ = task_model(imgs)
                loss.append(self.ce_loss(preds, labels))
        
        
        return sum(loss)/len(loader)




    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds,_ = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    
    
    def dis(self,p,p1,p2):
        dis=0
        m=nn.Softmax(1) 
        dis+=sum(sum(abs(m(p)-m(p1))))
        dis+=sum(sum(abs(m(p)-m(p2))))
        dis+=sum(sum(abs(m(p1)-m(p2))))
                
        dis=dis/self.args.num_classes/len(p[:,0])
        
        return dis
    
    
    def DLcal(self,f,f1,f2,loader):
        dis=0
        task_model1=copy.deepcopy(f)
        task_model2=copy.deepcopy(f)
        task_model1.linear=f1
        task_model2.linear=f2
        m=nn.Softmax(1) 
        f.eval()
        task_model1.eval()
        task_model2.eval()
        if self.args.cuda:
            f = f.cuda()
            task_model1 = task_model1.cuda()
            task_model2 = task_model2.cuda()
                    
        for imgs, _, _ in loader:
            imgs=imgs.cuda()
            with torch.no_grad():
                p=m(f(imgs))
                p1=m(task_model1(imgs))
                p2=m(task_model2(imgs))
            dis+=sum(sum(abs(p-p1)))
            dis+=sum(sum(abs(p-p2)))
            dis+=sum(sum(abs(p1-p2)))
            
       
        return dis
    
    def cross_entropy(self,input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))