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
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
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

    
    def train(self, querry_dataloader, val_dataloader, task_model,LN, unlabeled_dataloader,lr_input,mode,iter):
       
        R_task_model=copy.deepcopy(task_model)#####################################################
        R_LN=copy.deepcopy(LN)################################################################
        
        final_accuracy=0##########################################################################################################################################################
        
        self.args.train_iterations = int(self.args.global_iteration2*len(querry_dataloader)* self.args.train_epochs)
        #self.args.train_iterations = len(querry_dataloader)* self.args.train_epochs
        #self.args.train_iterations = (self.args.unlabeledbudget * self.args.train_epochs) // self.args.batch_size
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

                target_loss = self.ce_loss(preds, labels)
                task_loss=torch.sum(target_loss) / target_loss.size(0)
                
                optim_task_model.zero_grad()
                task_loss.backward()
                optim_task_model.step()
                if iter_count % 100 == 0:
                    print('Current training iteration: {}'.format(iter_count))
                    print('Current task model loss: {:.4f}'.format(task_loss.item()))
                    
                if iter_count % (len(querry_dataloader)* self.args.train_epochs) and task_loss.item()<0.002:##################################################################
                        if self.args.cuda:
                            task_model = task_model.cuda()
                            if self.test(task_model)>final_accuracy:
                                final_accuracy = max(final_accuracy,self.test(task_model))             
                                R_task_model=task_model #####################################################
                            
                        if self.validate(task_model, querry_dataloader)<0.001:
                            break#################################################################################
                
                
        elif mode==1:
            num_ftrs=task_model.linear.in_features
            task_model1=torch.nn.Linear(num_ftrs, self.args.num_classes)
            task_model2=torch.nn.Linear(num_ftrs, self.args.num_classes)
                
            LN = LN.train()
            optim_LN = optim.SGD(LN.parameters(), lr=lr_input, weight_decay=5e-4, momentum=0.9)
            
            
            
            for iter_count in range(self.args.train_iterations):
                labeled_imgs, labels = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)
                if iter_count % (len(querry_dataloader)* self.args.train_epochs) == 0:
                    for param in optim_task_model.param_groups:
                        param['lr'] = param['lr'] * self.args.lr_decay
                    for param in optim_LN.param_groups:
                        param['lr'] = param['lr'] * self.args.lr_decay

                if self.args.cuda:
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

                # task_model step
                preds, features =task_model(labeled_imgs)

                target_loss = self.ce_loss(preds, labels)
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
            
                pred_loss=LN(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                m_module_loss   = self.LossPredLoss(pred_loss, target_loss) 
                task_loss= m_backbone_loss +  m_module_loss
                
                
                
                optim_task_model.zero_grad()
                optim_LN.zero_grad()
                
                task_loss.backward()
                optim_task_model.step()
                optim_LN.step()
                
                
                
            
            
                if iter_count % 100 == 0:
                    print('Current training iteration: {}'.format(iter_count))
                    print('Current task model loss: {:.4f}'.format(m_backbone_loss.item()))
                    print('Current task model loss: {:.4f}'.format(m_module_loss.item()))
                        
                if iter_count % (len(querry_dataloader)* self.args.train_epochs) and m_backbone_loss.item()<0.002:##################################################################
                        if self.args.cuda:
                            task_model = task_model.cuda()
                            if self.test(task_model)>final_accuracy:
                                final_accuracy = max(final_accuracy,self.test(task_model))             
                                R_task_model=task_model #####################################################
                                R_LN=LN #################################################################_   
                        if self.validate(task_model, querry_dataloader)<0.001:
                            break#################################################################################
                
        
        
        
                


        print('acc: ', final_accuracy)
        return final_accuracy,R_task_model,R_LN
        








    def sample_for_labeling(self, task_model , LN, unlabeled_dataloader):
        querry_indices = self.sampler.sample(task_model,LN,unlabeled_dataloader,self.args.cuda,self.args.execute)

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
                target_loss=self.ce_loss(preds, labels)
                loss.append(torch.sum(target_loss) / target_loss.size(0))
                
        
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
    
    
    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss