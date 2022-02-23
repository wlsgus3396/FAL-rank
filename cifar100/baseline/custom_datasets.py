from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader 
import torchvision
import numpy
import torch
import os 
from PIL import Image
from utils import *



def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
       ])
    
    
def plain_cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
       ])


def core_cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
       ])

class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)
    
class plain_CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=plain_cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)
    

class core_CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=core_cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)

        
class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)


class plain_CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=plain_cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)

class core_CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=core_cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class ImageNet(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)




def mnist_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.286), (0.353))
       ])
    
    
def plain_mnist_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           transforms.Normalize((0.286), (0.353))
       ])
    
    
    
    
class MNIST(Dataset):
    def __init__(self, path):
        self.mnist = datasets.FashionMNIST(root=path,
                                        download=True,
                                        train=True,
                                        transform=mnist_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.mnist[index]

        return data, target, index

    def __len__(self):
        return len(self.mnist)
    
    
class plain_MNIST(Dataset):
    def __init__(self, path):
        self.mnist = datasets.FashionMNIST(root=path,
                                        download=True,
                                        train=True,
                                        transform=plain_mnist_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.mnist[index]

        return data, target, index

    def __len__(self):
        return len(self.mnist)
    
    
def collate_fn_A(batch):
    imgs, labels= zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)
    return torch.stack(imgs, dim=0), labels

class COVIDDataSet_binary_balance(torch.utils.data.Dataset):
    ''' Generate binary label > 0.5
        Make a balance between negative/positive samples
        All related info(filepath, annotation, cross validation set index, etc) is contained in csv file at each site '''

    def __init__(self, data_root, data_list, aug=False, balance=True, aug_params = [20, 0.1, 0.1],size=224):
        self.size=size
        self.data_root = data_root
        self.aug = aug
        self.balance = balance
        self.aug_params = aug_params
        assert os.path.exists(data_root), f"{data_root} NOT found."
        self.df=data_list
        self.label_list = list()
        self.path_list = list()

        self.sparse_col_name={}
        self._load_data() # load list of data path

        if balance:
            self._balance_data()

    def __len__(self): # indices are limited under this
        return len(self.label_list_final)

    def __getitem__(self, index):
        path = self.path_list_final[index]

        img_path = os.path.join(self.data_root, path) # path of one image
        label = self.label_list_final[index]
        img = Image.open(img_path).convert('RGB').resize((self.size,self.size))

        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img=transforms.functional.to_tensor(img)

        if self.aug:
            aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(self.aug_params[0],
                                        (self.aug_params[1], self.aug_params[1]),
                                        (1 - self.aug_params[2], 1 + self.aug_params[2])),
                transforms.ToTensor()
            ])
            img = aug_transform(img)
        img=normalize(img)

        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
            
        return img, label,index

    def _load_data(self):

        for idx, path in enumerate(self.df['Path']):

            label=self.df.loc[idx,"Score_72h"] # could be "Score_72h"
            label = (label>=0.5)*1 # make it as binary label

            self.label_list.append(label)
            self.path_list.append(path)

        if self.balance == False: # Validation case
            self.label_list_final = self.label_list
            self.path_list_final = self.path_list

    def _balance_data(self):
        ''' Balance positive/negative data by repeating positive '''
        import numpy as np

        p_idx = [i for i in range(len(self.label_list)) if self.label_list[i]==1]
        p_path_list = np.array(self.path_list)[p_idx]
        n_idx = [i for i in range(len(self.label_list)) if self.label_list[i]==0]
        n_path_list = np.array(self.path_list)[n_idx]

        ratio = len(n_path_list)//len(p_path_list)
        self.path_list_final =  list(p_path_list)*ratio + list(n_path_list)
        self.label_list_final = [1]*len(p_path_list)*ratio + [0]*len(n_path_list)
        
        
        
def noniid(dataset, num_users):
    import numpy as np

    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    num_shards, num_imgs = 400, 125       
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]


    # divide and assign
    idx_label=np.arange(10)
    for i in range(num_users):
        rand_set1=set(np.random.choice(idx_label, 2, replace=False)) # 기존것
        for rand in rand_set1:
            idx_label=list(set(idx_label) - set([rand]))  
        
        
        
        #rand_set1=set(random.sample([0,1,2,3,4,5,6,7,8,9],2))  # random seed 적용

        rand_set=[]
        for rand in rand_set1:  
            rand_set.extend(np.random.choice(list(set(np.arange(40*rand,40*(rand+1)))&set(idx_shard)), 20, replace=False))  #기존것
            #rand_set.extend(np.random.choice(np.arange(40*rand,40*(rand+1)),1))  # random seed 적용
        #rand_set= set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in set(rand_set):
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        
        
        rand_set=[]
        for rand in range(10):
            if rand not in list(rand_set1):
                rand_set.extend(np.random.choice(list(set(np.arange(40*rand,40*(rand+1)))&set(idx_shard)), 5, replace=False))  #기존것
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in set(rand_set):
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users