import os
import numpy as np
import scipy.io as scio

# The Pearson correlation coefficient
def PearsonCC(Y_gt, Y_pred):
    N = Y_gt.shape()[0]
    a = sum(Y_gt*Y_pred)-(sum(Y_gt)*sum(Y_pred))/N
    b = np.sqrt((sum(Y_gt*Y_gt)-(sum(Y_gt)*sum(Y_gt))/N)*(sum(Y_pred*Y_pred)-(sum(Y_pred)*sum(Y_pred))/N))
    Rpe = a/b
    return Rpe


# Reliability factor
def ZJR_factor(Y_gt, Y_pred):
    C = sum(Y_pred)/sum(Y_gt)
    dY_gt = np.diff(Y_gt) / 0.1
    ddY_gt = np.diff(Y_gt, 2) / 0.01
    dY_pred = np.diff(Y_pred) / 0.1
    ddY_pred = np.diff(Y_pred, 2) / 0.01
    F = np.abs(dY_gt - C*dY_pred)
    W = np.abs(ddY_gt - C*ddY_pred)/(np.abs(dY_gt)+np.abs(max(dY_gt)))
    Rp = np.sum(F*W)/np.sum(Y_gt)
    return Rp


def PR_factor(Y_gt, Y_pred):
    Rp = []
    LY_gt = (np.diff(Y_gt) / 0.1)/Y_gt
    LY_pred = (np.diff(Y_pred) / 0.1)/Y_pred
    YY_gt = LY_gt*LY_gt/LY_gt
    YY_pred = LY_pred*LY_pred/LY_pred
    Rp = np.sum((YY_gt-YY_pred)*(YY_gt-YY_pred))/np.sum(YY_gt*YY_gt+YY_pred*YY_pred)
    return Rp


# type='Zanazzi', 'Jona' , 'Pendry'
def R_factor(Y_gt, Y_pred, type='Zanazzi'):
    if type == 'Zanazzi':
        return ZJR_factor(Y_gt, Y_pred)
    elif type == 'Jona':
        return ZJR_factor(Y_gt, Y_pred)
    elif type == 'Pendry':
        return PR_factor(Y_gt, Y_pred)
    else:
        return np.array([ZJR_factor(Y_gt, Y_pred), PR_factor(Y_gt, Y_pred)])


def data_sperate_4_FSV(x):
    yf = np.fft.fft(x)
    #DC = yf[0:4]
    ib = 5
    N_2 = int(len(x)/2)
    for i in range(5, N_2):
        Lo = yf[4:i]
        Hi = yf[i:N_2]
        if np.sum(np.abs(Lo)) >= np.sum(np.abs(Hi))*0.4:
            ib = i
            x_ = np.zeros_like(x).astype(complex)
            DC = x_.copy()
            DC[0:4] = yf[0:4]
            LO = x_.copy()
            LO[4:ib] = yf[4:ib]
            HI = x_.copy()
            HI[ib:N_2] = yf[ib:N_2]
            dc = np.fft.ifft(DC)
            lo = np.fft.ifft(LO)
            hi = np.fft.ifft(HI)
            return dc, lo, hi
    return None, None, None


def FSV(Y_gt, Y_pred):
    dc_gt, lo_gt, hi_gt = data_sperate_4_FSV(Y_gt)
    dc_pred, lo_pred, hi_pred = data_sperate_4_FSV(Y_pred)
    N = dc_pred.shape[0]
    AMDi = np.abs((np.abs(dc_gt)-np.abs(dc_pred))/(np.sum((np.abs(dc_gt)+np.abs(dc_pred)))/N))
    ADM = np.sum(AMDi)/N
    FDM1i = (np.abs(np.diff(lo_gt, append=0)) - np.abs(np.diff(lo_pred, append=0))) / (np.sum(np.abs(np.diff(lo_gt, append=0)) + np.abs(np.diff(lo_pred, append=0))) *2/ N)
    FDM2i = (np.abs(np.diff(hi_gt, append=0)) - np.abs(np.diff(hi_pred, append=0))) / (np.sum(np.abs(np.diff(hi_gt, append=0)) + np.abs(np.diff(hi_pred, append=0))) *6/ N)
    FDM3i = (np.abs(np.diff(hi_gt, 2, prepend=0, append=0)) - np.abs(np.diff(hi_pred, 2, prepend=0, append=0))) / (np.sum(np.abs(np.diff(hi_gt, 2, prepend=0, append=0)) + np.abs(np.diff(hi_pred, 2, prepend=0, append=0))) *7.2/ N)
    FDMi = 2*np.abs(FDM1i+FDM2i+FDM3i)
    FDM = np.sum(FDMi)/N
    GDMi = np.sqrt(AMDi*AMDi+FDMi*FDMi)
    GDM = np.sum(GDMi)/N
    return ADM, FDM, GDM


if __name__ == '__main__':
    filename = 'farfeild_antenna_1_with_head_912MHz_LT.mat'
    dat = scio.loadmat(filename)
    AIE = dat['AIE']
    E2D = dat['E2D']
    ADMs=[]
    FDMs=[]
    GDMs=[]
    for i in range(33):
        Y_gt = np.squeeze(E2D[i, :])
        Y_pred = np.squeeze(AIE[i, :])
        ADM, FDM, GDM = FSV(Y_gt, Y_pred)
        ADMs.append(ADM)
        FDMs.append(FDM)
        GDMs.append(GDM)
    scio.savemat(filename, {'AIE': AIE, 'E2D': E2D, 'Phi': dat['Phi'], 'Theta': dat['Theta'], 'ADMs':  np.array(ADMs), 'FDMs':  np.array(FDMs), 'GDMs': np.array(GDMs)})




