import numpy as np
from torch.utils.data.dataset import Dataset
import os
import cv2
import torch
from PIL import Image
import numpy
import scipy.io as scio


def get_files(file_path, suffix='.mat'):
    files=[]
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            if f.endswith(suffix):
                files.append(os.path.join(file_path, f))

    return files


def standard_dataset(files):
    SARs = np.zeros((len(files), 12))
    TISs = np.zeros((len(files), 7))
    EIRPs = np.zeros((len(files), 7, 2112))
    for i, file in enumerate(files):
        dat = scio.loadmat(file)
        SARs[i, :] = dat['SARs']
        SARs[i, :] = dat['SARs']
        tis = np.array([dat['freespace_TIS'], dat['LHand_TIS'], dat['RHand_TIS'], dat['LCHead_TIS'], dat['RCHead_TIS'], dat['LHH_TIS'], dat['RHH_TIS']]).squeeze()

        eirp = np.concatenate([dat['freespace_EIRP'].reshape(1, 2112),
                                dat['LHand_EIRP'].reshape(1, 2112),
                                dat['RHand_EIRP'].reshape(1, 2112),
                                dat['LCHead_EIRP'].reshape(1, 2112),
                                dat['RCHead_EIRP'].reshape(1, 2112),
                                dat['LHH_EIRP'].reshape(1, 2112),
                                dat['RHH_EIRP'].reshape(1, 2112)], axis=0).squeeze()
        TISs[i, :] = tis
        EIRPs[i, :, :] = eirp
    sar_mean = np.mean(SARs, axis=0)
    sar_std = np.std(SARs, axis=0)
    tis_mean = np.mean(TISs, axis=0)
    tis_std = np.std(TISs, axis=0)
    eirp_mean = np.mean(EIRPs, axis=0)
    eirp_std = np.std(EIRPs, axis=0)
    return sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std


class EMFDateset(Dataset):
    def __init__(self, filelist, sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std):
        self.dataset = filelist
        self.sar_mean, self.sar_std, self.tis_mean, self.tis_std, self.eirp_mean, self.eirp_std = sar_mean, sar_std, tis_mean, tis_std, eirp_mean, eirp_std

    def __getitem__(self, index):
        #print(self.dataset[index])
        dat = scio.loadmat(self.dataset[index])
        '''
        {'ID': ant_id, 'fre': frequency, 'S11': S11,
                  'SARs': np.array([LC_WBSAR, LC_1gSAR, LC_10SAR, RC_WBSAR, RC_1gSAR, RC_10SAR, LT_WBSAR, LT_1gSAR, LT_10SAR, RT_WBSAR, RT_1gSAR, RT_10SAR]),
                  'Jx': Jx, 'Jy': Jy, 'Jz': Jz, 'J': J,
                  'theta': theta, 'phi': phi,
                  'freespace_EIRP': freespace_EIRP, 'freespace_TRP': freespace_TRP,'freespace_TIS': freespace_TIS,
                  'RCHead_EIRP': RCHead_EIRP, 'RCHead_TRP': RCHead_TRP,'RCHead_TIS': RCHead_TIS,
                  'LCHead_EIRP': LCHead_EIRP, 'LCHead_TRP': LCHead_TRP,'LCHead_TIS': LCHead_TIS,
                  'LHand_EIRP': LHand_EIRP, 'LHand_TRP': LHand_TRP,'LHand_TIS': LHand_TIS,
                  'RHand_EIRP': RHand_EIRP, 'RHand_TRP': RHand_TRP,'RHand_TIS': RHand_TIS,
                  'LHH_EIRP': LHH_EIRP, 'LHH_TRP': LHH_TRP,'LHH_TIS': LHH_TIS,
                  'RHH_EIRP': RHH_EIRP, 'RHH_TRP': RHH_TRP,'RHH_TIS': RHH_TIS})
        '''
        fre = dat['fre'].squeeze(0)
        # J = dat['J'][5:75, 3:23, 5:105]
        # Jx = dat['Jx'][:, 5:75, 3:23, 5:105]
        # Jy = dat['Jy'][:, 5:75, 3:23, 5:105]
        # Jz = dat['Jz'][:, 5:75, 3:23, 5:105]
        # Jxyz = np.concatenate([Jx, Jy, Jz], axis=0)
        # J = np.transpose(J, (1, 0, 2))
        # Jxyz = np.transpose(Jxyz, (0, 2, 1, 3))

        Ex = dat['Ex']
        Ey = dat['Ey']
        Ez = dat['Ez']
        # EX = Ex[0,:,:,:]*Ex[0,:,:,:]+Ex[1,:,:,:]*Ex[1,:,:,:]
        # EY = Ey[0,:,:,:]*Ey[0,:,:,:]+Ey[1,:,:,:]*Ey[1,:,:,:]
        # EZ = Ez[0,:,:,:]*Ez[0,:,:,:]+Ez[1,:,:,:]*Ez[1,:,:,:]
        Exyz = np.concatenate([Ex, Ey, Ez], axis=0)
        # Exyz = np.concatenate([np.expand_dims(EX, axis=0), np.expand_dims(EY, axis=0), np.expand_dims(EZ, axis=0)], axis=0)

        Hx = dat['Hx']
        Hy = dat['Hy']
        Hz = dat['Hz']
        # HX = Hx[0,:,:,:]*Hx[0,:,:,:]+Hx[1,:,:,:]*Hx[1,:,:,:]
        # HY = Hy[0,:,:,:]*Hy[0,:,:,:]+Hy[1,:,:,:]*Hy[1,:,:,:]
        # HZ = Hz[0,:,:,:]*Hz[0,:,:,:]+Hz[1,:,:,:]*Hz[1,:,:,:]
        Hxyz = np.concatenate([Hx, Hy, Hz], axis=0)
        #Hxyz = np.concatenate([np.expand_dims(HX, axis=0), np.expand_dims(HY, axis=0), np.expand_dims(HZ, axis=0)], axis=0)
        #H = np.transpose(H, (1, 0, 2))
        EHxyz = np.concatenate([Exyz, Hxyz], axis=0)

        SARs = dat['SARs'] # 1x12
        SARs = (SARs-self.sar_mean)/self.sar_std
        SARs = SARs.squeeze()
        '[tis_fs, tis_Lhand, tis_Rhand, tis_Lhead, tis_Rhead, tis_Lhh, tis_Rhh],\
               [eirp_fs, eirp_Lhand, eirp_Rhand, eirp_Lhead, eirp_Rhead, eirp_Lhh, eirp_Rhh]'
        TISs = np.array([dat['freespace_TIS'], dat['LHand_TIS'], dat['RHand_TIS'], dat['LCHead_TIS'], dat['RCHead_TIS'], dat['LHH_TIS'], dat['RHH_TIS']]).squeeze()
        TISs = (TISs-self.tis_mean)/self.tis_std
        TRPs = np.array([dat['freespace_TRP'], dat['LHand_TRP'], dat['RHand_TRP'], dat['LCHead_TRP'], dat['RCHead_TRP'], dat['LHH_TRP'], dat['RHH_TRP']]).squeeze()

        EIRPs = np.concatenate([dat['freespace_EIRP'].reshape(1, 2112),
                                dat['LHand_EIRP'].reshape(1, 2112),
                                dat['RHand_EIRP'].reshape(1, 2112),
                                dat['LCHead_EIRP'].reshape(1, 2112),
                                dat['RCHead_EIRP'].reshape(1, 2112),
                                dat['LHH_EIRP'].reshape(1, 2112),
                                dat['RHH_EIRP'].reshape(1, 2112)], axis=0).squeeze()
        EIRPs = (EIRPs-self.eirp_mean)/self.eirp_std

        EHxyz = EHxyz[:,0:-1:4,0:-1:4,0:-1:4]

        return (fre-600)/(6000-60), torch.from_numpy(EHxyz), torch.from_numpy(SARs),\
               torch.from_numpy(TISs), torch.from_numpy(TRPs), torch.from_numpy(EIRPs)

    def __len__(self):
        return len(self.dataset)

