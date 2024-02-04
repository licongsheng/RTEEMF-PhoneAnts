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
resume = './checkpoints/RTEMF_resnet_10.pth'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def train_model(vis, model, optimizer, criterion, dataloader, epoch, batch_size, evaluator,
                train_losses, train_metrices):
    tot_sar_loss=0
    tot_tis_loss=0
    tot_eirp_loss=0
    tot_sar_acc=0
    tot_tis_acc=0
    tot_eirp_acc=0
    train_loss = 0.0
    model.train()
    tbar = tqdm(dataloader)
    for i, (fre, Jxyz, SARs, TISs, TRPs, EIRPs) in enumerate(tbar):
        J_dev, SARs_dev, TISs_dev, TRPs_dev, EIRPs_dev = Jxyz.cuda().float(), SARs.cuda().float(), TISs.cuda().float(), TRPs.cuda().float(), EIRPs.cuda().float()
        optimizer.zero_grad()
        sar, tiss, eirps = model(J_dev, fre.cuda().float())
        sar_loss = criterion(sar, SARs_dev)
        tis_loss = criterion(tiss, TISs_dev)
        #trp_loss = criterion(torch.sum(eirps), torch.sum(EIRPs_dev))
        eirp_loss = criterion(eirps, EIRPs_dev)
        loss = sar_loss+tis_loss + eirp_loss
        #loss = tis_loss + eirp_loss
        tot_sar_loss = tot_sar_loss+sar_loss.item()
        tot_tis_loss = tot_tis_loss+tis_loss.item()
        tot_eirp_loss = tot_eirp_loss+eirp_loss.item()

        sar_acc, tis_acc, eirp_acc = evaluator.add_batch(SARs_dev, TISs_dev, TRPs_dev, EIRPs_dev, sar, tiss, eirps)
        tot_sar_acc = tot_sar_acc+sar_acc
        tot_tis_acc = tot_tis_acc+tis_acc
        tot_eirp_acc = tot_eirp_acc+eirp_acc
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f, SAR Loss: %.3f, TIS Loss: %.3f, EIRP Loss: %.3f' % (loss.item(), sar_loss.item(), tis_loss.item(), eirp_loss.item()))
        # vis.heatmap(a_image, win=orig_image)
        # vis.heatmap(a_target, win=gt_image)
        # vis.heatmap(a_pred, win=pred_image)
        #
        # if (epoch == 0 or epoch == 10) and i == 0:
        #     vis.line(X=np.column_stack((np.array([i]),
        #                                 np.array([i]),
        #                                 np.array([i]))),
        #              Y=np.column_stack((np.array([sar_loss.item()]),
        #                                 np.array([tis_loss.item()]),
        #                                 np.array([eirp_loss.item()]))),
        #              update='replace', win=train_losses,
        #              opts=dict(legend=['SAR Loss', 'TIS Loss', 'EIRP Loss']))
        #
        #     vis.line(X=np.column_stack((np.array([i]),
        #                                 np.array([i]),
        #                                 np.array([i]))),
        #              Y=np.column_stack((np.array([sar_acc]),
        #                                 np.array([tis_acc]),
        #                                 np.array([eirp_acc]))),
        #              update='replace', win=train_metrices,
        #              opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))
        # else:
        #     vis.line(X=np.column_stack((np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]))),
        #              Y=np.column_stack((np.array([sar_loss.item()]),
        #                                 np.array([tis_loss.item()]),
        #                                 np.array([eirp_loss.item()]))),
        #              update='append', win=train_losses,
        #              opts=dict(legend=['SAR Loss', 'TIS Loss', 'EIRP Loss']))
        #
        #     vis.line(X=np.column_stack((np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]))),
        #              Y=np.column_stack((np.array([sar_acc]),
        #                                 np.array([tis_acc]),
        #                                 np.array([eirp_acc]))),
        #              update='append', win=train_metrices,
        #              opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))

    print('[Epoch: %d, Train numImages: %5d]' % (epoch, i * batch_size))
    print('Loss: %.3f' % (train_loss/(i * batch_size)))
    return (model, tot_sar_loss/(i * batch_size), tot_tis_loss/(i * batch_size), tot_eirp_loss/(i * batch_size),
            tot_sar_acc/(i * batch_size), tot_tis_acc/(i * batch_size), tot_eirp_acc/(i * batch_size))


def evaluate_model(vis, model, dataloader, epoch, batch_size, evaluator, val_metrices):
    val_loss = 0.0
    tot_sar_acc=0
    tot_tis_acc=0
    tot_eirp_acc=0
    model.eval()
    #evaluator.reset()
    tbar = tqdm(dataloader)
    num_img_tr = len(dataloader)
    for i, (fre, Jxyz, SARs, TISs, TRPs, EIRPs) in enumerate(tbar):
        J_dev, SARs_dev, TISs_dev, TRPs_dev, EIRPs_dev = Jxyz.cuda().float(), SARs.cuda().float(), TISs.cuda().float(), TRPs.cuda().float(), EIRPs.cuda().float()
        optimizer.zero_grad()
        sar, tiss, eirps = model(J_dev, fre.cuda().float())

        sar_acc, tis_acc, eirp_acc = evaluator.add_batch(SARs_dev, TISs_dev, TRPs_dev, EIRPs_dev, sar, tiss, eirps)

        tot_sar_acc = tot_sar_acc+sar_acc
        tot_tis_acc = tot_tis_acc+tis_acc
        tot_eirp_acc = tot_eirp_acc+eirp_acc

        tbar.set_description('Value SAR Acc: %.3f, TIS Acc: %.3f, EIRP Acc: %.3f' % (sar_acc, tis_acc, eirp_acc))

        # vis.heatmap(a_image, win=orig_image)
        # vis.heatmap(a_target, win=gt_image)
        # vis.heatmap(a_pred, win=pred_image)
        # if (epoch == 0 or epoch == 10) and i == 0:
        #     vis.line(X=np.column_stack((np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]))),
        #              Y=np.column_stack((np.array([sar_acc]),
        #                                 np.array([tis_acc]),
        #                                 np.array([eirp_acc]))),
        #              update='replace', win=val_metrices,
        #              opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))
        # else:
        #     vis.line(X=np.column_stack((np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]),
        #                                 np.array([epoch*len(dataloader)+i]))),
        #              Y=np.column_stack((np.array([sar_acc]),
        #                                 np.array([tis_acc]),
        #                                 np.array([eirp_acc]))),
        #              update='append', win=val_metrices,
        #              opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))

    print('[Epoch: %d, evaluation numImages: %5d]' % (epoch, num_img_tr * batch_size))
    return tot_sar_acc/(i * batch_size), tot_tis_acc/(i * batch_size), tot_eirp_acc/(i * batch_size)

def saveVisdomData(win, env, filename, mode='w'):
    viz = visdom.Visdom()
    win_data = viz.get_window_data(win, env)
    pre_data = json.loads(win_data)
    x = pre_data["content"]["data"][0]["x"]
    sar = pre_data["content"]["data"][0]["y"]
    tis = pre_data["content"]["data"][1]["y"]
    eirp = pre_data["content"]["data"][2]["y"]
    with open(filename, mode) as f:
        writer = csv.writer(f)
        for i in range(len(x)):
            writer.writerow([x[i], sar[i], tis[i], eirp[i]])


if __name__ == "__main__":
    epoches = 300
    '''model_depth in [10, 18, 34, 50, 101, 152, 200]'''
    model_depth = 18
    checkpoint_name_prefix = './checkpoints/RTEMF_4_resnet_'+str(model_depth)
    vis = visdom.Visdom(env='RTEAntenna_4_ResNet_'+str(model_depth))
    gpu_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = get_files('./Datasets')
    #files = files[0:4]
    print(files)
    sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std = standard_dataset(files)
    train_files = files
    random.shuffle(files)
    val_files = files[1:5]
    torch_dataset = EMFDateset(train_files, sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std)
    training_data = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    torch_dataset = EMFDateset(val_files, sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std)
    testing_data = DataLoader(torch_dataset, batch_size=batch_size)

    '''n_input_channels = 1 for magnitude or 6 for complex'''
    model = RTEEMFAntPerformace(model_depth=model_depth, n_input_channels=12,  n_features=384) # 5760: 10 18 34 ; 23040: 50 # 1x1x0.2: 960 2x2x0.5: 384
    model.apply(weights_init_normal)
    # model.require_grad_model_params(backbone=False, sar=False, fs_ota=True, lhead_ota=True, rhead_ota=True,
    #                        lhand_ots=True, rhand_ots=True, lhh_ota=True, rhh_ota=True)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    # if resume:
    #     if os.path.isfile(resume):
    #         print("=> loading checkpoint '{}'".format(resume))
    #         checkpoint = torch.load(resume)
    #         model.load_state_dict(checkpoint)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(resume))

    criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    evaluator_train = Evaluator()
    evaluator_val = Evaluator()
    train_losses = vis.line(X=np.column_stack((np.array(0), np.array(0), np.array(0))),
                              Y=np.column_stack((np.array(0), np.array(0), np.array(0))))
    train_metrices = vis.line(X=np.column_stack((np.array(0), np.array(0), np.array(0))),
                              Y=np.column_stack((np.array(0), np.array(0), np.array(0))))
    val_metrices = vis.line(X=np.column_stack((np.array(0), np.array(0), np.array(0))),
                            Y=np.column_stack((np.array(0), np.array(0), np.array(0))))
    print(model)
    best_pred = 0
    for epoch in range(epoches):
        model, tot_sar_loss, tot_tis_loss, tot_eirp_loss, tot_sar_acc, tot_tis_acc, tot_eirp_acc =\
            train_model(vis, model, optimizer, criterion, training_data, epoch, batch_size, evaluator_train,
                            train_losses, train_metrices)
        val_sar_acc, val_tis_acc, val_eirp_acc = evaluate_model(vis, model, testing_data, epoch, batch_size, evaluator_val,
                            val_metrices)

        if epoch == 0:
            vis.line(X=np.column_stack((np.array([epoch]),
                                            np.array([epoch]),
                                            np.array([epoch]))),
                         Y=np.column_stack((np.array([tot_sar_loss]),
                                            np.array([tot_tis_loss]),
                                            np.array([tot_eirp_loss]))),
                         update='replace', win=train_losses,
                         opts=dict(legend=['SAR Loss', 'TIS Loss', 'EIRP Loss']))
            vis.line(X=np.column_stack((np.array([epoch]),
                                            np.array([epoch]),
                                            np.array([epoch]))),
                         Y=np.column_stack((np.array([tot_sar_acc]),
                                            np.array([tot_tis_acc]),
                                            np.array([tot_eirp_acc]))),
                         update='replace', win=train_metrices,
                         opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))

            vis.line(X=np.column_stack((np.array([epoch]),
                                        np.array([epoch]),
                                        np.array([epoch]))),
                     Y=np.column_stack((np.array([val_sar_acc]),
                                        np.array([val_tis_acc]),
                                        np.array([val_eirp_acc]))),
                     update='replace', win=val_metrices,
                     opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))
        else:
            vis.line(X=np.column_stack((np.array([epoch]),
                                        np.array([epoch]),
                                        np.array([epoch]))),
                     Y=np.column_stack((np.array([tot_sar_loss]),
                                        np.array([tot_tis_loss]),
                                        np.array([tot_eirp_loss]))),
                     update='append', win=train_losses,
                     opts=dict(legend=['SAR Loss', 'TIS Loss', 'EIRP Loss']))

            vis.line(X=np.column_stack((np.array([epoch]),
                                        np.array([epoch]),
                                        np.array([epoch]))),
                     Y=np.column_stack((np.array([tot_sar_acc]),
                                        np.array([tot_tis_acc]),
                                        np.array([tot_eirp_acc]))),
                     update='append', win=train_metrices,
                     opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))

            vis.line(X=np.column_stack((np.array([epoch]),
                                        np.array([epoch]),
                                        np.array([epoch]))),
                     Y=np.column_stack((np.array([val_sar_acc]),
                                        np.array([val_tis_acc]),
                                        np.array([val_eirp_acc]))),
                     update='append', win=val_metrices,
                     opts=dict(legend=['SAR Acc', 'TIS Acc', 'EIRP Acc']))
        #torch.save(model.state_dict(), checkpoint_name_prefix + '_%d.pth' % (epoch))
        torch.save(model.state_dict(), checkpoint_name_prefix + '.pth')

    saveVisdomData(train_losses, "RTEAntenna_4_ResNet_"+str(model_depth), "RTEAntenna_4_ResNet_"+str(model_depth)+"_train_loss.csv")
    saveVisdomData(train_metrices, "RTEAntenna_4_ResNet_"+str(model_depth), "RTEAntenna_4_ResNet_"+str(model_depth)+"_train_acc.csv")
    saveVisdomData(val_metrices, "RTEAntenna_4_ResNet_"+str(model_depth), "RTEAntenna_4_ResNet_"+str(model_depth)+"_val_acc.csv")