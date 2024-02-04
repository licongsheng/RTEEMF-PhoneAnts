import numpy as np
import torch

import torch
class Evaluator(object):
    def __init__(self):
        self.SAR_acc = []
        self.TRP_acc = []
        self.TIS_acc = []
        self.EIRP_acc = []

        self.gt_sar=[]
        self.gt_sar=[]
        self.gt_tis=[]
        self.gt_trp=[]
        self.gt_eirp=[]
        self.pred_sar=[]
        self.pred_tis=[]
        self.pred_trp=[]
        self.pred_eirp=[]

        self.criterion = torch.nn.MSELoss(size_average=True)

    def add_batch(self, gt_sar, gt_tis, gt_trp, gt_eirp, pred_sar, pred_tis, pred_eirp):
        self.gt_sar=gt_sar
        self.gt_tis=gt_tis
        self.gt_trp=gt_eirp
        self.gt_eirp=gt_eirp
        self.pred_sar=pred_sar
        self.pred_tis=pred_tis
        self.pred_eirp=pred_eirp
        self.pred_trp=[]
        sar_acc = 1-self.criterion(gt_sar, pred_sar).item()
        #sar_acc = 1-torch.mean(torch.abs((pred_sar-gt_sar)/gt_sar)).item()
        tis_acc = 1-self.criterion(gt_tis, pred_tis).item()
        #tis_acc = 1-torch.mean(torch.abs((pred_tis-gt_tis)/gt_tis)).item()
        eirp_acc = 1-self.criterion(gt_eirp, pred_eirp).item()
        #eirp_acc = 1-torch.mean(torch.abs((pred_eirp-gt_eirp)/gt_eirp)).item()
        self.SAR_acc.append(sar_acc)
        self.TIS_acc.append(tis_acc)
        self.EIRP_acc.append(eirp_acc)

        return sar_acc, tis_acc, eirp_acc

    def reset(self):
        self.SAR_acc = []
        self.TRP_acc = []
        self.TIS_acc = []
        self.EIRP_acc = []

        self.gt_sar = []
        self.gt_sar = []
        self.gt_tis = []
        self.gt_trp = []
        self.gt_eirp = []
        self.pred_sar = []
        self.pred_tis = []
        self.pred_trp = []
        self.pred_eirp = []
