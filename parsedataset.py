import os
import numpy as np
import scipy.io as scio
import pandas as pd
import json
from scipy import interpolate
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
#from cupyx.scipy.interpolate import RegularGridInterpolator

def loadexcel(filename):
    # 使用pandas读取excel文件
    ex_data = pd.read_excel(filename)
    # 读取excel中表头
    head_list = list(ex_data.columns)
    list_dict = []
    # 组装json格式数据并保存到列表中
    for i in ex_data.values:
        # 使用表头和每行的数据组装json格式数据
        a_line = dict(zip(head_list, i))
        list_dict.append(a_line)
    return list_dict


def load_J_mat_148(filename):
    dat = scio.loadmat(filename)
    J = dat['RMS_Modulus_of_Vector_0s']
    x_axis = dat['x_axis']
    y_axis = dat['y_axis']
    z_axis = dat['z_axis']
    Jxr = dat['Real_Part_of_X_Comp_0s']
    Jyr = dat['Real_Part_of_Y_Comp_0s']
    Jzr = dat['Real_Part_of_Z_Comp_0s']
    Jxi = dat['Imaginary_Part_of_X_Comp_0s']
    Jyi = dat['Imaginary_Part_of_Y_Comp_0s']
    Jzi = dat['Imaginary_Part_of_Z_Comp_0s']
    Jx = np.concatenate([np.expand_dims(Jxr, 0), np.expand_dims(Jxi, 0)])
    Jy = np.concatenate([np.expand_dims(Jyr, 0), np.expand_dims(Jyi, 0)])
    Jz = np.concatenate([np.expand_dims(Jzr, 0), np.expand_dims(Jzi, 0)])

    return Jx, Jy, Jz, J

def load_EH_mat_148(filename):
    dat = scio.loadmat(filename.replace('Surface J', 'E_field'))
    E = dat['RMS_Modulus_of_Vector_0s']
    x_axis = dat['x_axis']
    y_axis = dat['y_axis']
    z_axis = dat['z_axis']
    Exr = dat['Real_Part_of_X_Comp_0s']
    Eyr = dat['Real_Part_of_Y_Comp_0s']
    Ezr = dat['Real_Part_of_Z_Comp_0s']
    Exi = dat['Imaginary_Part_of_X_Comp_0s']
    Eyi = dat['Imaginary_Part_of_Y_Comp_0s']
    Ezi = dat['Imaginary_Part_of_Z_Comp_0s']
    # Exr = resample_NEMF(x_axis, y_axis, z_axis, Exr)
    # Eyr = resample_NEMF(x_axis, y_axis, z_axis, Eyr)
    # Ezr = resample_NEMF(x_axis, y_axis, z_axis, Ezr)
    # Exi = resample_NEMF(x_axis, y_axis, z_axis, Exi)
    # Eyi = resample_NEMF(x_axis, y_axis, z_axis, Eyi)
    # Ezi = resample_NEMF(x_axis, y_axis, z_axis, Ezi)

    Ex = np.concatenate([np.expand_dims(Exr, 0), np.expand_dims(Exi, 0)])
    Ey = np.concatenate([np.expand_dims(Eyr, 0), np.expand_dims(Eyi, 0)])
    Ez = np.concatenate([np.expand_dims(Ezr, 0), np.expand_dims(Ezi, 0)])
    dat = scio.loadmat(filename.replace('Surface J', 'H_field'))
    H = dat['RMS_Modulus_of_Vector_0s']
    x_axis = dat['x_axis']
    y_axis = dat['y_axis']
    z_axis = dat['z_axis']
    Hxr = dat['Real_Part_of_X_Comp_0s']
    Hyr = dat['Real_Part_of_Y_Comp_0s']
    Hzr = dat['Real_Part_of_Z_Comp_0s']
    Hxi = dat['Imaginary_Part_of_X_Comp_0s']
    Hyi = dat['Imaginary_Part_of_Y_Comp_0s']
    Hzi = dat['Imaginary_Part_of_Z_Comp_0s']
    # Hxr = resample_NEMF(x_axis, y_axis, z_axis, Hxr)
    # Hyr = resample_NEMF(x_axis, y_axis, z_axis, Hyr)
    # Hzr = resample_NEMF(x_axis, y_axis, z_axis, Hzr)
    # Hxi = resample_NEMF(x_axis, y_axis, z_axis, Hxi)
    # Hyi = resample_NEMF(x_axis, y_axis, z_axis, Hyi)
    # Hzi = resample_NEMF(x_axis, y_axis, z_axis, Hzi)
    Hx = np.concatenate([np.expand_dims(Hxr, 0), np.expand_dims(Hxi, 0)])
    Hy = np.concatenate([np.expand_dims(Hyr, 0), np.expand_dims(Hyi, 0)])
    Hz = np.concatenate([np.expand_dims(Hzr, 0), np.expand_dims(Hzi, 0)])

    return Ex, Ey, Ez, Hx, Hy, Hz,x_axis,y_axis,z_axis


def load_farfield_148(filename):
    theta = []
    phi = []
    Etot = []
    filename = filename.replace('\\\u202a', '/')
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != "%" and len(line) > 1:
                strs = line.split('\t\t')
                Etot.append(float(strs[0]))
                theta.append(float(strs[1])/180*np.pi)
                phi.append(float(strs[2])/180*np.pi)
    return theta, phi, Etot


def resample_NEMF(x_axis, y_axis, z_axis, volume):
    nx = int(70e-3/5e-4)+1
    ny = int(6e-3/1e-4)+1
    nz = int(135e-3/5e-4)+1
    interpolator = RegularGridInterpolator((x_axis, y_axis, z_axis), volume, method='linear') # method = “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”.
    vol = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                vol[i, j, k] = interpolator([-0.035+i*5e-4, -0.006+j*1e-4, -0.13+k*5e-4])
    return vol


def export_data2mat(ant_dict):
    ant_id = ant_dict['Antenna ID']
    j_file = ant_dict['Surface J mat file']
    emf_path = os.path.dirname(os.path.abspath(j_file))
    frequency = ant_dict['Frequency [MHz]']
    S11 = ant_dict['S-Parameters (dB)']

    '''SAR'''
    LC_WBSAR = ant_dict['Mean SAR Left Cheek']
    LC_1gSAR = ant_dict['Left Cheek Peak 1g SAR']
    LC_10SAR = ant_dict['Left Cheek Peak 10g SAR']
    RC_WBSAR = ant_dict['Mean SAR Right Cheek']
    RC_1gSAR = ant_dict['Right Cheek Peak 1g SAR']
    RC_10SAR = ant_dict['Right Cheek Peak 10g SAR']
    LT_WBSAR = ant_dict['Mean SAR Left Tilt']
    LT_1gSAR = ant_dict['Left Tilt Peak 1g SAR']
    LT_10SAR = ant_dict['Left Tilt Peak 10g SAR']
    RT_WBSAR = ant_dict['Mean SAR Right Tilt']
    RT_1gSAR = ant_dict['Right Tilt Peak 1g SAR']
    RT_10SAR = ant_dict['Right Tilt Peak 10g SAR']

    '''OTA'''
    freespace_TRP = ant_dict['TRP in free space']
    freespace_TIS = ant_dict['TIS in free space']
    freespace_EIRP_file = ant_dict['EIRP in free space']

    LCHead_TRP = ant_dict['TRP with Head only(LC)']
    LCHead_TIS = ant_dict['TIS with Head only(LC)']
    LCHead_EIRP_file = ant_dict['EIRP  with Head only(LC)']

    LTHead_TRP = ant_dict['TRP with Head only(LT)']
    LTHead_TIS = ant_dict['TIS with Head only(LT)']
    LTHead_EIRP_file = ant_dict['EIRP with Head only(LT)']

    RCHead_TRP = ant_dict['TRP with Head only(RC)']
    RCHead_TIS = ant_dict['TIS with Head only(RC)']
    RCHead_EIRP_file = ant_dict['EIRP with Head only(RC)']

    RTHead_TRP = ant_dict['TRP with Head only(RT)']
    RTHead_TIS = ant_dict['TIS with Head only(RT)']
    RTHead_EIRP_file = ant_dict['EIRP with Head only(RT)']
    RTHead_EIRP_file = RTHead_EIRP_file.replace('Right cheek', 'Right Tilt')

    LHand_TRP = ant_dict['TRP with Hand Only']
    LHand_TIS = ant_dict['TIS with Hand Only']
    LHand_EIRP_file = ant_dict['EIRP with Hand Only']

    RHand_TRP = LTHead_TRP
    RHand_TIS = LTHead_TIS
    RHand_EIRP_file = LTHead_EIRP_file

    LHH_TRP = ant_dict['TRP with Head *Hand']
    LHH_TIS = ant_dict['TIS with Head *Hand']
    LHH_EIRP_file = ant_dict['EIRP with Head *Hand']

    RHH_TRP = RTHead_TRP
    RHH_TIS = RTHead_TIS
    RHH_EIRP_file = RTHead_EIRP_file

    '''
    load data from external file ... 
    '''
    Jx, Jy, Jz, J = load_J_mat_148(os.path.join('../Datasets', j_file))
    Ex, Ey, Ez, Hx, Hy, Hz,x_axis,y_axis,z_axis = load_EH_mat_148(os.path.join('../Datasets', j_file))
    theta, phi, freespace_EIRP = load_farfield_148(os.path.join('../Datasets', freespace_EIRP_file))
    _, _, LCHead_EIRP = load_farfield_148(os.path.join('../Datasets', LCHead_EIRP_file))
    print(RCHead_EIRP_file)
    _, _, RCHead_EIRP = load_farfield_148(os.path.join('../Datasets', RCHead_EIRP_file))
    _, _, LHand_EIRP = load_farfield_148(os.path.join('../Datasets', LHand_EIRP_file))
    _, _, RHand_EIRP = load_farfield_148(os.path.join('../Datasets', RHand_EIRP_file))
    _, _, LHH_EIRP = load_farfield_148(os.path.join('../Datasets', LHH_EIRP_file))
    _, _, RHH_EIRP = load_farfield_148(os.path.join('../Datasets', RHH_EIRP_file))
    mat_filename = './Datasets/Antenna_'+str(ant_id) + '_' + str(int(frequency)) +'MHz.mat'
    print('Saving ' + mat_filename + '...\n')
    scio.savemat(mat_filename,
                 {'ID': ant_id, 'fre': frequency, 'S11': S11,
                  'SARs': np.array([LC_WBSAR, LC_1gSAR, LC_10SAR, RC_WBSAR, RC_1gSAR, RC_10SAR, LT_WBSAR, LT_1gSAR, LT_10SAR, RT_WBSAR, RT_1gSAR, RT_10SAR]),
                  'Jx': Jx, 'y_axis': y_axis, 'z_axis': z_axis,
                  'x_axis': x_axis, 'Jy': Jy, 'Jz': Jz,
                  'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
                  'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
                  'theta': theta, 'phi': phi,
                  'freespace_EIRP': freespace_EIRP, 'freespace_TRP': freespace_TRP,'freespace_TIS': freespace_TIS,
                  'RCHead_EIRP': RCHead_EIRP, 'RCHead_TRP': RCHead_TRP,'RCHead_TIS': RCHead_TIS,
                  'LCHead_EIRP': LCHead_EIRP, 'LCHead_TRP': LCHead_TRP,'LCHead_TIS': LCHead_TIS,
                  'LHand_EIRP': LHand_EIRP, 'LHand_TRP': LHand_TRP,'LHand_TIS': LHand_TIS,
                  'RHand_EIRP': RHand_EIRP, 'RHand_TRP': RHand_TRP,'RHand_TIS': RHand_TIS,
                  'LHH_EIRP': LHH_EIRP, 'LHH_TRP': LHH_TRP,'LHH_TIS': LHH_TIS,
                  'RHH_EIRP': RHH_EIRP, 'RHH_TRP': RHH_TRP,'RHH_TIS': RHH_TIS})


if __name__ == '__main__':
    xls_filename = r'../Datasets/Dataset.xlsx'
    data = loadexcel(xls_filename)
    for ant_dic in data:
        export_data2mat(ant_dic)