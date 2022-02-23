import torch
from torch.distributions import Categorical
import torch.nn as nn
import random
import numpy as np
import copy
from seqsampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from torch.utils.data import DataLoader

class AdversarySampler:
    def __init__(self, args):
        self.args=args
        
        


    def sample(self, task_model,FC1, FC2, data, querry_dataloader,data_unlabeled,subset,labeled_set,labeled_data_size, cuda,execute):
       
            
        if self.args.execute=='RANDOM':
            
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
        
        elif self.args.execute=='coreset' or self.args.execute=='F-coreset':
            
            unlabeled_loader = DataLoader(data_unlabeled, batch_size=self.args.batch_size, 
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)
            task_model.eval()
            features = torch.tensor([]).cuda()

            with torch.no_grad():
                for inputs, _, _ in unlabeled_loader:
                    
                    inputs = inputs.cuda()
                    _,features_batch = task_model(inputs)
                    features = torch.cat((features, features_batch), 0)
                
                feat = features.detach().cpu().numpy()
                new_av_idx = np.arange(len(subset),(len(subset) + labeled_data_size))
                sampling = kCenterGreedy(feat)  
                batch = sampling.select_batch_(new_av_idx, int(labeled_data_size*self.args.budget))##############################
                other_idx = [x for x in range(len(subset)) if x not in batch]
            
            
            
            
            return other_idx + batch
        
        
        