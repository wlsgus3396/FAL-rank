import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--gpu', type=int, default=1, help='GPU server')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')

    parser.add_argument('--num_clients', type=int, default=5, help='number of clinets')
    parser.add_argument('--initial_budget', type=int, default=1000, help='initial budget')
    parser.add_argument('--budget', type=int, default=1000, help='budget')
    parser.add_argument('--unlabeledbudget', type=int, default=10000, help='unlabeledbudget')
    parser.add_argument('--K', type=int, default=5, help='experiment')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_solo', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.997, help='learning rate')

    parser.add_argument('--global_iteration1', type=int, default=9, help='global iteration')
    parser.add_argument('--global_iteration2', type=int, default=10000, help='global iteration')
    parser.add_argument('--train_epochs', type=int, default=1, help='Number of training epochs')




    parser.add_argument('--execute', type=str, default='RANDOM' , help='strategy')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=1, help='Number of discrepancy steps taken for every task model step')
    ### num_dis_steps might be 2 as in VAAL
    
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
