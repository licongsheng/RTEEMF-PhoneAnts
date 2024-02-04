from pathlib import Path
import json
import random
import os
import csv
import numpy as np
import torch
import visdom

from dataload import *
from models import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
batch_size = 1
resume = './checkpoints/rms_RTEMF_4_resnet_18.pth'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


def evaluate_model(model, dataloader, criterion):
    model.eval()
    tbar = tqdm(dataloader)
    num_img_tr = len(dataloader)
    for i, (fre, Jxyz, SARs, TISs, TRPs, EIRPs) in enumerate(tbar):
        J_dev, SARs_dev, TISs_dev, TRPs_dev, EIRPs_dev = Jxyz.cuda().float(), SARs.cuda().float(), TISs.cuda().float(), TRPs.cuda().float(), EIRPs.cuda().float()
        sar, tiss, eirps = model(J_dev, fre.cuda().float())
        sar_loss = criterion(sar, SARs_dev)
        tis_loss = criterion(tiss, TISs_dev)
        eirp_loss = criterion(eirps, EIRPs_dev)
        
        tbar.set_description('Value SAR ACC: %.3f, TIS Acc: %.3f, EIRP Acc: %.3f' % (1-sar_loss.item(), 1-tis_loss.item(), 1-eirp_loss.item()))
        
        scio.savemat('./res/val_dataset_'+str(i)+'.mat',{'fre':fre.numpy(),
                                                         'GT_SAR': SARs.numpy(), 'GT_TIS': TISs.numpy(), 'GT_TRP': TRPs.numpy(), 'GT_EIRPs': EIRPs.numpy(),
                                                         'AI_SAR': sar.cpu().detach().numpy(), 'AI_TIS': tiss.cpu().detach().numpy(), 'AI_EIRPs': eirps.cpu().detach().numpy()})

if __name__ == "__main__":
    epoches = 300
    '''model_depth in [10, 18, 34, 50, 101, 152, 200]'''
    model_depth = 18
    gpu_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = get_files('./Datasets')
    print(files)
    sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std = standard_dataset(files)
    
    scio.savemat("./res/statisticinfos.mat", {'mean_sar': sar_mean, 'std_sar': sar_std,
                                              'mean_tis': tis_mean, 'std_tis': tis_std,
                                              'mean_eirp': eirp_mean, 'std_eirp': eirp_std})
    train_files = files
    torch_dataset = EMFDateset(files, sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std)
    test_data = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    '''n_input_channels = 1 for magnitude or 6 for complex'''
    model = RTEEMFAntPerformace(model_depth=model_depth, n_input_channels=12,  n_features=384) # 5760: 10 18 34 ; 23040: 50 # 1x1x0.2: 960 2x2x0.5: 384
    model.apply(weights_init_normal)
    # model.require_grad_model_params(backbone=False, sar=False, fs_ota=True, lhead_ota=True, rhead_ota=True,
    #                        lhand_ots=True, rhand_ots=True, lhh_ota=True, rhh_ota=True)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    criterion = nn.MSELoss(size_average=True)

    evaluate_model(model, test_data, criterion)

