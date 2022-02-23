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

    
    def train(self, querry_dataloader, val_dataloader, task_model,vae,discriminator, unlabeled_dataloader,lr_input,mode,iter):
       
        
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
        
    
        
    
        if mode==0:
            num_ftrs=task_model.linear.in_features
            task_model1=torch.nn.Linear(num_ftrs, self.args.num_classes)
            task_model2=torch.nn.Linear(num_ftrs, self.args.num_classes)
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
                    
                    
        elif mode==1:
            num_ftrs=task_model.linear.in_features
            task_model1=torch.nn.Linear(num_ftrs, self.args.num_classes)
            task_model2=torch.nn.Linear(num_ftrs, self.args.num_classes)
            optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
            
            vae.train()
            discriminator.train()
            
            if self.args.cuda:
                vae = vae.cuda()
                discriminator = discriminator.cuda()
            
            
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
                
                
                
                    # VAE step
                for count in range(self.args.num_vae_steps):
                    recon, z, mu, logvar = vae(labeled_imgs)
                    unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                    unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                    transductive_loss = self.vae_loss(unlabeled_imgs, 
                            unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
                
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                        
                    if self.args.cuda:
                        lab_real_preds = lab_real_preds.cuda()
                        unlab_real_preds = unlab_real_preds.cuda()

                    dsc_loss = self.bce_loss(np.squeeze(labeled_preds), np.squeeze(lab_real_preds)) + \
                            self.bce_loss(np.squeeze(unlabeled_preds), np.squeeze(unlab_real_preds))
                    total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                    optim_vae.zero_grad()
                    total_vae_loss.backward()
                    optim_vae.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_vae_steps - 1):
                        labeled_imgs, _ = next(labeled_data)
                        unlabeled_imgs = next(unlabeled_data)

                        if self.args.cuda:
                            labeled_imgs = labeled_imgs.cuda()
                            unlabeled_imgs = unlabeled_imgs.cuda()
                            labels = labels.cuda()

                # Discriminator step
                for count in range(self.args.num_adv_steps):
                    with torch.no_grad():
                        _, _, mu, _ = vae(labeled_imgs)
                        _, _, unlab_mu, _ = vae(unlabeled_imgs)
                    
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                    if self.args.cuda:
                        lab_real_preds = lab_real_preds.cuda()
                        unlab_fake_preds = unlab_fake_preds.cuda()
                    
                    dsc_loss = self.bce_loss(np.squeeze(labeled_preds), np.squeeze(lab_real_preds)) + \
                            self.bce_loss(np.squeeze(unlabeled_preds), np.squeeze(unlab_fake_preds))

                    optim_discriminator.zero_grad()
                    dsc_loss.backward()
                    optim_discriminator.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_adv_steps - 1):
                        labeled_imgs, _ = next(labeled_data)
                        unlabeled_imgs = next(unlabeled_data)

                        if self.args.cuda:
                            labeled_imgs = labeled_imgs.cuda()
                            unlabeled_imgs = unlabeled_imgs.cuda()
                            labels = labels.cuda()
                
                
            
            
                if iter_count % 100 == 0:
                    print('Current training iteration: {}'.format(iter_count))
                    print('Current task model loss: {:.4f}'.format(task_loss.item()))
                    print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                    print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))
            
        
                

                
                        
            
        elif mode==2:
                
            optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
            
            vae.train()
            discriminator.train()
            
            if self.args.cuda:
                vae = vae.cuda()
                discriminator = discriminator.cuda()
            
            
            for iter_count in range(self.args.train_iterations):
                labeled_imgs, labels = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)
                

                if self.args.cuda:
                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

                
                
                    # VAE step
                for count in range(self.args.num_vae_steps):
                    recon, z, mu, logvar = vae(labeled_imgs)
                    unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                    unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                    transductive_loss = self.vae_loss(unlabeled_imgs, 
                            unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
                
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                        
                    if self.args.cuda:
                        lab_real_preds = lab_real_preds.cuda()
                        unlab_real_preds = unlab_real_preds.cuda()

                    dsc_loss = self.bce_loss(np.squeeze(labeled_preds), np.squeeze(lab_real_preds)) + \
                            self.bce_loss(np.squeeze(unlabeled_preds), np.squeeze(unlab_real_preds))
                    total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                    optim_vae.zero_grad()
                    total_vae_loss.backward()
                    optim_vae.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_vae_steps - 1):
                        labeled_imgs, _ = next(labeled_data)
                        unlabeled_imgs = next(unlabeled_data)

                        if self.args.cuda:
                            labeled_imgs = labeled_imgs.cuda()
                            unlabeled_imgs = unlabeled_imgs.cuda()
                            labels = labels.cuda()

                # Discriminator step
                for count in range(self.args.num_adv_steps):
                    with torch.no_grad():
                        _, _, mu, _ = vae(labeled_imgs)
                        _, _, unlab_mu, _ = vae(unlabeled_imgs)
                    
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                    if self.args.cuda:
                        lab_real_preds = lab_real_preds.cuda()
                        unlab_fake_preds = unlab_fake_preds.cuda()
                    
                    dsc_loss = self.bce_loss(np.squeeze(labeled_preds), np.squeeze(lab_real_preds)) + \
                            self.bce_loss(np.squeeze(unlabeled_preds), np.squeeze(unlab_fake_preds))

                    optim_discriminator.zero_grad()
                    dsc_loss.backward()
                    optim_discriminator.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_adv_steps - 1):
                        labeled_imgs, _ = next(labeled_data)
                        unlabeled_imgs = next(unlabeled_data)

                        if self.args.cuda:
                            labeled_imgs = labeled_imgs.cuda()
                            unlabeled_imgs = unlabeled_imgs.cuda()
                            labels = labels.cuda()
                    
                
            
            
                if iter_count % 100 == 0:
                    print('Current training iteration: {}'.format(iter_count))
                    print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                    print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))
            
           
                
                
        
                
        final_accuracy=0
        stop=1
        if task_loss.item()<0.001:
            if self.args.cuda:
                task_model = task_model.cuda()                
            if self.validate(task_model, val_dataloader)<0.0005:
                stop=0
                final_accuracy = self.test(task_model)
                print('acc: ', final_accuracy)
            else:
                final_accuracy=0
                stop=1
        return final_accuracy,task_model,vae,discriminator,stop
        




    def sample_for_labeling(self, task_model,vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(task_model,vae, discriminator,
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
                loss.append(self.ce_loss(preds, labels))
        
        
        return sum(loss)/len(loader)




    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)
                preds= torch.round(torch.sigmoid(preds)).cpu()
            #preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += (np.squeeze(preds) == labels).sum().item()
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
    
    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD