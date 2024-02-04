import os
import numpy as np
import scipy.io as scio
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtk.util.vtkConstants import *
import cv2 as cv

def load_J_mat_148(filename):
    dat = scio.loadmat(filename)
    J = dat['Real_Modulus_of_Vector_0s']
    pts = dat['Points']
    cells = dat['Cells']
    ncells = int(cells.shape[1]/4)
    cells = np.reshape(cells, (ncells, 4))
    Jx = J[:, 0]
    Jy = J[:, 1]
    Jz = J[:, 2]
    J = np.sqrt(abs(Jx)*abs(Jx)+abs(Jy)*abs(Jy)+abs(Jz)*abs(Jz))

    vtk_pts = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    for i in range(pts.shape[0]):
        vtk_pts.InsertNextPoint(pts[i])
        scalars.InsertNextTuple1(np.log(np.max(J[i])))

    for i in range(ncells):
        ids = cells[i, 1:]
        vtk_cells.InsertNextCell(3, ids)

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(vtk_cells)
    pd.GetPointData().SetScalars(scalars)

    lut = vtk.vtkLookupTable()
    lut.SetTableRange(-10, np.log(np.max(J)))
    lut.SetHueRange(0.667, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()

    return J, pd, lut


def load_J_mat_192(filename):
    dat = scio.loadmat(filename)
    J = dat['Snapshot0']
    pts = dat['Points']
    cells = dat['Cells']
    ncells = int(cells.shape[1]/4)
    cells = np.reshape(cells, (ncells, 4))
    Jx = J[:, 0]
    Jy = J[:, 1]
    Jz = J[:, 2]
    J = np.sqrt(abs(Jx)*abs(Jx)+abs(Jy)*abs(Jy)+abs(Jz)*abs(Jz))

    vtk_pts = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    for i in range(pts.shape[0]):
        vtk_pts.InsertNextPoint(pts[i])
        scalars.InsertNextTuple1(np.log(np.max(J[i])))

    for i in range(ncells):
        ids = cells[i, 1:]
        vtk_cells.InsertNextCell(3, ids)

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(vtk_cells)
    pd.GetPointData().SetScalars(scalars)

    lut = vtk.vtkLookupTable()
    lut.SetTableRange(-10, np.log(np.max(J)))
    lut.SetHueRange(0.667, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()

    return J, pd, lut


def load_farfield_148(filename):
    theta = []
    phi = []
    Etot = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != "%" and len(line) > 1:
                strs = line.split('\t\t')
                Etot.append(float(strs[0]))
                theta.append(float(strs[1])/180*np.pi)
                phi.append(float(strs[2])/180*np.pi)
    return theta, phi, Etot


def load_farfield_192(filename):
    theta = []
    phi = []
    Etheta = []
    Ephi = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != "#" and len(line) > 1:
                strs = line.split('\t')
                theta.append(float(strs[1]))
                phi.append(float(strs[2]))
                Etheta.append(complex(float(strs[3]), float(strs[4])))
                Ephi.append(complex(float(strs[5]), float(strs[6])))
    return theta, phi, Etheta, Ephi


def render_polydata(pd, lut=None):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    if lut is not None:
        mapper.ScalarVisibilityOn()
        mapper.SetUseLookupTableScalarRange(256)
        mapper.SetLookupTable(lut)
    mapper.Update()

    color_bar_Actor = vtk.vtkScalarBarActor()
    color_bar_Actor.SetLookupTable(mapper.GetLookupTable())
    color_bar_Actor.SetWidth(0.1)
    color_bar_Actor.SetTitle("Gain")
    color_bar_Actor.GetTitleTextProperty().SetColor(0,0,0)
    color_bar_Actor.SetNumberOfLabels(6)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.AddActor(color_bar_Actor)
    #ren.SetBackground(0.1, 0.2, 0.4)
    ren.SetBackground(1, 1, 1)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(100)

    myCamera = ren.GetActiveCamera()
    myCamera.SetViewUp(0, 0, 1)
    myCamera.SetPosition(0, 0, 50)
    #myCamera.SetFocalPoint(0, 0, 0)
    myCamera.ComputeViewPlaneNormal()
    #myCamera.Azimuth(90.0)
    #myCamera.Elevation(00.0)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()


def render_farfield_148(theta, phi, Etot):
    nGain = len(theta)
    N_th = len(np.unique(theta))
    N_phi = len(np.unique(phi))
    E_2D = np.zeros((N_th, N_phi)).astype(np.float64)

    for i in range(N_th):
        for j in range(N_phi):
            E_2D[i][j] = Etot[i*N_phi+j]

    scio.savemat('farfield2D.mat', {'E2D': E_2D,'Theta': np.unique(theta),'Phi': np.unique(phi)})
    vtk_pts = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    Etotal = []
    for i in range(nGain):
        gain = Etot[i]
        Etotal.append(gain)
        rcoselev = gain*np.cos(np.pi/2-theta[i])
        point = np.array([rcoselev*np.cos(phi[i]), rcoselev*np.sin(phi[i]), gain*np.sin(np.pi/2-theta[i])])
        vtk_pts.InsertNextPoint(point)
        scalars.InsertNextTuple1(np.log(gain))

    for i in range(N_th-1):
        for j in range(N_phi):
            vtk_cells.InsertNextCell(4, [i*N_phi+j, i*N_phi+j+1, (i+1)*N_phi+j+1, (i+1)*N_phi+j])

    for j in range(N_phi):
        vtk_cells.InsertNextCell(4, [(N_th-1) * N_phi + j, (N_th-1) * N_phi + j + 1, 0 * N_phi + j + 1, 0 * N_phi + j])
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(vtk_cells)
    pd.GetPointData().SetScalars(scalars)

    lut = vtk.vtkLookupTable()
    lut.SetTableRange(np.log(np.min(Etotal)), np.log(np.max(Etotal)))
    lut.SetHueRange(0.667, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()

    render_polydata(pd, lut)


def render_farfield_192(theta, phi, Etheta, Ephi):
    nGain = len(theta)
    N_th = len(np.unique(theta))
    N_phi = len(np.unique(phi))
    vtk_pts = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    Etotal = []
    for i in range(nGain):
        gain = np.sqrt(abs(Etheta[i])*abs(Etheta[i])+abs(Ephi[i])*abs(Ephi[i]))
        Etotal.append(gain)
        rcoselev = gain*np.cos(np.pi/2-theta[i])
        point = np.array([rcoselev*np.cos(phi[i]), rcoselev*np.sin(phi[i]), gain*np.sin(np.pi/2-theta[i])])
        vtk_pts.InsertNextPoint(point)
        scalars.InsertNextTuple1(np.log(gain))

    for i in range(N_th-1):
        for j in range(N_phi):
            vtk_cells.InsertNextCell(4, [i*N_phi+j, i*N_phi+j+1, (i+1)*N_phi+j+1, (i+1)*N_phi+j])
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(vtk_cells)
    pd.GetPointData().SetScalars(scalars)

    lut = vtk.vtkLookupTable()
    lut.SetTableRange(np.log(np.min(Etotal)), np.log(np.max(Etotal)))
    lut.SetHueRange(0.667, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()

    render_polydata(pd, lut)



if __name__ == '__main__':
    j_filename = '../Datasets_V14/Antenna1/free space/Surface J mat file_912MHz.mat'
    farfield_filename = '../Datasets_V14/Antenna2/Head+Hand/antena_1_1428MHz_head+hand.txt'
    #J, pd, lut = load_J_mat_148(j_filename)
    theta, phi, Etot = load_farfield_148(farfield_filename)
    render_farfield_148(theta, phi, np.array(Etot))
    #theta, phi, Etheta, Ephi = load_farfield_192(farfield_filename)
    #render_farfield(theta, phi, np.array(Etheta), np.array(Ephi))
    #render_polydata(pd, lut)