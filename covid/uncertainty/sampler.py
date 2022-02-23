import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy

class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model,FC1, FC2, DL_F, DL_I, data, cuda,execute):
        if self.args.execute=='F-dis' or self.args.execute=='dis':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()

               
                task_model1=copy.deepcopy(task_model)
                task_model2=copy.deepcopy(task_model)
                task_model1.linear=FC1
                task_model2.linear=FC2
                task_model.eval()
                task_model1.eval()
                task_model2.eval()    
                if cuda:
                    task_model = task_model.cuda()
                    task_model1 = task_model1.cuda()
                    task_model2 = task_model2.cuda()
                with torch.no_grad():    
                    p=m(task_model(images))
                    p1=m(task_model1(images))
                    p2=m(task_model2(images))
                    
                for i in range(len(indices)):
                    
                    preds.append(abs(sum(abs(p[i,:]-p1[i,:]))/len(p[0,:])+sum(abs(p[i,:]-p2[i,:]))/len(p[0,:]) + sum(abs(p1[i,:]-p2[i,:]))/len(p[0,:])-DL_I))    

                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk 
            
           
            
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.args.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
            
        elif self.args.execute=='RANDOM' or self.args.execute=='RANDOM2':
            
            all_indices = []

            for images, _, indices in data:
                all_indices.extend(indices)
            querry_indices=random.sample(range(len(all_indices)),self.args.budget)
            querry_pool_indices = np.asarray(all_indices)[querry_indices]
            
            return querry_pool_indices

        elif self.args.execute=='uncertainty' or self.args.execute=='F-uncertainty':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()

            
                task_model.eval()
                    
                if cuda:
                    task_model = task_model.cuda()
                    
                with torch.no_grad():    
                    p=m(task_model(images))
                    
                    
                for i in range(len(indices)):
                    preds.append(Categorical(probs = p[i,:]).entropy())    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(len(all_indices)*self.args.budget))##############################
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        
        
        elif self.args.execute=='MCdrop-entropy' or self.args.execute=='F-MCdrop-entropy':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()

            
                task_model.eval()
                    
                if cuda:
                    task_model = task_model.cuda()
                    
                with torch.no_grad():    
                    p=m(task_model(images))
                    for _ in range(24):
                        p+=m(task_model(images))
                p=p/25
                    
                    
                for i in range(len(indices)):
                    preds.append(Categorical(probs = p[i,:]).entropy())    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.args.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        elif self.args.execute=='MCdrop-bald' or self.args.execute=='F-MCdrop-bald':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()

            
                task_model.eval()
                    
                if cuda:
                    task_model = task_model.cuda()
                    
                
                    
                phat=[]
                phat2=[]
                with torch.no_grad():
                    phat=m(task_model(images))
                    for _ in range(24):
                        phat+=m(task_model(images))
                    for _ in range(25):    
                        phat2.append(m(task_model(images)))    
                phat=phat/25
                
                
                for i in range(len(indices)):
                        
                    tphat=phat2[0]
                    
                    Chat=Categorical(probs = tphat[i,:]).entropy()
                    for ii in range(24):
                        tphat=phat2[ii]        
                        Chat+=Categorical(probs = tphat[i,:]).entropy()
                    
                    Chat=Chat/25
                    
                
                    
                    preds.append(Categorical(probs = phat[i,:]).entropy()-Chat)    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.args.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        elif self.args.execute=='MCdrop-VAR' or self.args.execute=='F-MCdrop-VAR':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()

            
                task_model.eval()
                    
                if cuda:
                    task_model = task_model.cuda()
                    
                
                phat=[]
                
                with torch.no_grad():
                    for _ in range(25):    
                        phat.append(m(task_model(images)))
                
                
                
                for i in range(len(indices)):
                
                    aa=torch.zeros(self.args.num_classes)
                    
                    for ii in range(25):        
                        #p=torch.tensor(phat[ii], device='cpu')
                        a=torch.argmax(phat[ii][i])    
                        aa[a]+=1
                    n=torch.max(aa)        
                    preds.append(1-n/25)    
                
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.args.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
        
        
        
        
        
        elif self.args.execute=='F-LL' or self.args.execute=='LL':
            all_preds = []
            all_indices = []
            m=nn.Softmax(1)   
            for images, _, indices in data:
                preds=[]
                if cuda:
                    images = images.cuda()
                task_model.eval()
                task_model=task_model.cuda()
                LN.eval()
                LN=LN.cuda()
                
                
                with torch.no_grad():    
                    scores,features=task_model(images)
                    p=LN(features)
                    p = p.view(p.size(0))
                    
                for i in range(len(indices)):
                    if execute=="LL":
                        preds.append(p[i])    
                    elif execute=='F-LL':
                        preds.append(p[i])
                        
                        
                        
                preds = torch.tensor(preds,device='cpu')
                all_preds.extend(preds)
                all_indices.extend(indices)

            all_preds = torch.stack(all_preds)
            all_preds = all_preds.view(-1)
            # need to multiply by -1 to be able to use torch.topk 
        
            
            # select the points which the discriminator things are the most likely to be unlabeled
            _, querry_indices = torch.topk(all_preds, int(self.args.budget))
            querry_pool_indices = np.asarray(all_indices)[querry_indices]

            return querry_pool_indices
