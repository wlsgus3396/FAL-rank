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






class Solver:
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
       
        if mode==0:
            thr1=0.001
            thr2=0.0005
        elif mode==1:
            thr1=0.002
            thr2=0.001
        self.args.train_iterations = len(querry_dataloader)* self.args.train_epochs
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
        
    
        
    
        
        num_ftrs=task_model.linear.in_features
        task_model1=torch.nn.Linear(num_ftrs, self.args.num_classes)
        task_model2=torch.nn.Linear(num_ftrs, self.args.num_classes)
        
        
        task_model1.load_state_dict(FC1.state_dict())
        task_model2.load_state_dict(FC2.state_dict())
        for iter_count in range(self.args.train_iterations):
            

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds=task_model(labeled_imgs)

            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()
            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))

        final_accuracy=0
        stop=1
        if task_loss.item()<thr1:
            if self.args.cuda:
                task_model = task_model.cuda()                
            if self.validate(task_model, val_dataloader)<thr2:
                stop=0
                final_accuracy = self.test(task_model)
                print('acc: ', final_accuracy)
            else:
                final_accuracy=0
                stop=1
        
        return final_accuracy,task_model,FC1, FC2, stop
        




    def sample_for_labeling(self, task_model,FC1, FC2, DL_dis, DL_item, unlabeled_dataloader):
        querry_indices = self.sampler.sample(task_model,FC1, FC2, DL_dis,DL_item,
                                             unlabeled_dataloader, 
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
                preds = task_model(imgs)
                loss.append(self.bcelogit_loss(np.squeeze(preds), labels.float()))
        
        
        return sum(loss)/len(loader)




    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)
                preds= torch.round(torch.sigmoid(preds))
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
    
    def append_dropout(self,model, rate=0.7):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout(module)
            if isinstance(module, nn.ReLU):
                new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
                setattr(model, name, new)