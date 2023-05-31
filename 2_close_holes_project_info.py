"""
    Copyright (c) - Marta Nunez Garcia
    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
    any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    Public License for more details. You should have received a copy of the GNU General Public License along with this
    program. If not, see <http://www.gnu.org/licenses/>.
"""

"""
    Close holes corresponding to PVs and LAA. Mark filled holes with a scalar array. Additionally, transfer all scalar arrays from input mesh to output (closed) mesh
    Hole filling done with implementation from https://github.com/cbutakoff/tools/tree/master/FillSurfaceHoles  Related publication: P. Liepa "Filling Holes in Meshes", 2003.
    Hole filling can also be done manually with reMESH (http://remesh.sourceforge.net/)

    Input: mesh with PVs and LAA clipped at their ostia + same mesh without MV
    Output: mesh with holes corresponding to PVs and LAA filled and marked with scalar array. No MV.
    Usage: python 2_close_holes_project_info.py --meshfile_open data/mesh_crinkle_clipped.vtk --meshfile_open_no_mitral  data/mesh_clipped_mitral.vtk --meshfile_closed data/mesh_clipped_c.vtk
"""

from aux_functions import *
import sys
import os
import argparse
from sys import platform

parser = argparse.ArgumentParser()
parser.add_argument('--meshfile_open', type=str, metavar='PATH', help='path to input mesh with clipped PVs and LAA')
parser.add_argument('--meshfile_open_no_mitral', type=str, metavar='PATH', help='path to input mesh with additional MV clip')
parser.add_argument('--meshfile_closed', type=str, metavar='PATH', help='path to output mesh, i.e. with filled holes')
args = parser.parse_args()




################################################
################################################
################################################

args.meshfile = meshfile
args.meshfile_open = meshfile_open
args.meshfile_open_no_mitral = meshfile_open_no_mitral
args.meshfile_closed = meshfile_closed

################################################
################################################
################################################





fileroot = os.path.dirname(args.meshfile_open)
filename = os.path.basename(args.meshfile_open)
filenameroot = os.path.splitext(filename)[0]

# if not os.path.exists(args.meshfile_open):
#     sys.exit('ERROR: Input file 1 (LA after PV, LAA clipping) not found')
# if not os.path.exists(args.meshfile_open_no_mitral):
#     sys.exit('ERROR: Input file 2 (LA after PV, LAA, and MV clipping) not found')
# if os.path.exists(args.meshfile_closed):
#     print('WARNING: Closed mesh already exists. Delete it and run again if you want to update it.')
# else:  # Fill holes
#     if platform == "linux" or platform == "linux2":
#         # os.system('./FillSurfaceHoles -i ' + args.meshfile_open + ' -o ' + args.meshfile_closed)
#         if (vtk.vtkVersion.GetVTKVersion() < '9.1.0'):
#             os.system('./FillSurfaceHoles_old -i ' + args.meshfile_open + ' -o ' + args.meshfile_closed + ' -smooth none')
#         else:
#             os.system('./FillSurfaceHoles -i ' + args.meshfile_open + ' -o ' + args.meshfile_closed + ' -smooth none')
#     elif platform == "win32":
#         os.system('FillSurfaceHoles_Windows/FillSurfaceHoles.exe -i ' + args.meshfile_open + ' -o ' + args.meshfile_closed + ' -smooth none')   # default smooth cotangent (and edglen) fails when using the Windows binary
#     else:
#         sys.exit('Unknown operating system. Holes cannot be filled automatically. Fill holes manually and save file as ', args.meshfile_closed, '. Then run again this script to proyect scalar arrays from initial mesh if necessary.')



import os 
import pyvista as pv 

def vtk2stl(From, To): 
    stl_f = pv.read(os.path.join(From))
    pv.save_meshio(To, stl_f)
    print('changed to stl !')




# def surface_reconstruction(points):
#     point_cloud = vtk.vtkPolyData()
#     point_cloud.SetPoints(points)

#     # Poisson surface reconstruction
#     poisson = vtk.vtkPoissonReconstruction()
#     poisson.SetInputData(point_cloud)
#     poisson.Update()

#     # Get reconstructed surface
#     reconstructed_surface = poisson.GetOutput()
#     return reconstructed_surface


# mesh = readvtk(args.meshfile_open)
# surface_reconstruction(mesh.GetPoints())
# """
# png_visualize1(mesh)
# """
# surface_reconstruction



import pyvista as pv
import numpy as np





# surface = readvtk(args.meshfile_open)


surface = readvtk(args.meshfile)
surface = convert_cell_data_to_point_data(surface)
m_open = readvtk(args.meshfile_open)
"""
png_visualize1(surface)
png_visualize1(m_open)
png_mesh_GetArray(m_open)
png_mesh_GetArray(surface)
"""


# print('Projecting information... ')
transfer_all_scalar_arrays(surface, m_open)


writevtk(m_open, args.meshfile_open[:-4]+"-zone.vtk")

args.meshfile_open = args.meshfile_open[:-4]+"-zone.vtk"

#



png_writestl(args.meshfile_open, args.meshfile_open[:-4]+".stl")
surface_reconst(args.meshfile_open[:-4]+".stl", args.meshfile_closed[:-4]+".stl", depth=7)
writevtk(png_readstl(args.meshfile_closed[:-4]+".stl"), args.meshfile_closed)


# Example usage
# input_stl_file = output_mesh_file_stl
# output_vtk_file = output_mesh_file_vtk
# resolution = 1000
# convert_stl_to_structured_vtk(input_stl_file, output_vtk_file[:-4]+".vtk", resolution)



# surface = readvtk(args.meshfile_open)
# survace_fill = fillholes(surface, size=1000)
# writevtk(survace_fill, args.meshfile_closed)



m_open = readvtk(args.meshfile_open)
m_no_mitral = readvtk(args.meshfile_open_no_mitral)
m_closed = readvtk(args.meshfile_closed)



print('Projecting information... ')
transfer_all_scalar_arrays(m_open, m_closed)

# m_closed2 = pointthreshold(m_closed, 'zone', .001, 2.5)

# m_closed = fillholes(m_closed, holesize=100)

"""
png_visualize1(m_closed)
png_visualize1(m_closed2)
png_visualize1(m_open)

png_mesh_GetArray(m_open)
png_mesh_GetArray(m_closed)

png_mesh_GetCoord(m_open).shape
png_mesh_GetCoord(m_closed).shape

png_mesh_GetCoord(m_open)
png_mesh_GetCoord(m_closed)

writevtk(m_open, 'E:/data/map/flat/3218-CUVIA-cut-poly/tmp_open.vtk')
writevtk(m_closed, 'E:/data/map/flat/3218-CUVIA-cut-poly/tmp_closed.vtk')

png_mesh_GetArray2(m_open, 'x')
png_mesh_GetArray2(m_closed, 'x')

"""



"""
png_visualize1(m_open)
png_visualize1(m_closed)


png_mesh_GetArray(m_closed)


np.concatenate((png_mesh_GetArray2(m_closed,'x')[:,np.newaxis],\
               png_mesh_GetArray2(m_closed,'y')[:,np.newaxis],\
               png_mesh_GetArray2(m_closed,'z')[:,np.newaxis]), axis=1 )

np.concatenate((png_mesh_GetArray2(m_open,'x')[:,np.newaxis],\
               png_mesh_GetArray2(m_open,'y')[:,np.newaxis],\
               png_mesh_GetArray2(m_open,'z')[:,np.newaxis]), axis=1 )

tmp_open=np.concatenate((png_mesh_GetArray2(m_open,'x')[:,np.newaxis],\
               png_mesh_GetArray2(m_open,'y')[:,np.newaxis],\
               png_mesh_GetArray2(m_open,'z')[:,np.newaxis]), axis=1 )
tmp_closed=np.concatenate((png_mesh_GetArray2(m_closed,'x')[:,np.newaxis],\
               png_mesh_GetArray2(m_closed,'y')[:,np.newaxis],\
               png_mesh_GetArray2(m_closed,'z')[:,np.newaxis]), axis=1 )


pd.DataFrame(tmp_open).duplicated().sum()
pd.DataFrame(tmp_open).shape[0]
pd.DataFrame(tmp_closed).shape[0] - pd.DataFrame(tmp_closed).duplicated().sum()


"""


dist_list = np.zeros(m_closed.GetNumberOfPoints())

# Mark filled holes. Common points (close enough, not added during hole filling) will we marked with scalar array
array_labelsA = np.zeros(m_closed.GetNumberOfPoints())
locator = vtk.vtkPointLocator()
locator.SetDataSet(m_open)
locator.BuildLocator()
for p in range(m_closed.GetNumberOfPoints()):
    point = m_closed.GetPoint(p)
    closestpoint_id = locator.FindClosestPoint(point)
    
    """
    point
    m_open.GetPoint(closestpoint_id)
    """
    dist = euclideandistance(point, m_open.GetPoint(closestpoint_id))
    dist_list[p] = dist
    
    if dist > 4:   # empirical distance: 3, 2, 4
        array_labelsA[p] = 1
newarray = numpy_to_vtk(array_labelsA)
newarray.SetName('hole')
m_closed.GetPointData().AddArray(newarray)

"""
np.histogram( dist_list )
m_final1 = pointthreshold(m_closed, 'hole', 0, 0)
png_visualize1(m_closed)
png_visualize1(m_final1)
"""


# Mark MV using m_open and m_no_mitral
array_labelsB = np.zeros(m_open.GetNumberOfPoints())
locator = vtk.vtkPointLocator()
locator.SetDataSet(m_no_mitral)
locator.BuildLocator()
for p in range(m_open.GetNumberOfPoints()):
    point = m_open.GetPoint(p)
    closestpoint_id = locator.FindClosestPoint(point)
    dist = euclideandistance(point, m_no_mitral.GetPoint(closestpoint_id))
    if dist > 0.1:   # empirical distance
        array_labelsB[p] = 1
newarray = numpy_to_vtk(array_labelsB)
newarray.SetName('mv')
m_open.GetPointData().AddArray(newarray)
"""
png_visualize1(m_open)
png_visualize1(m_closed)
"""

transfer_array(m_open, m_closed, 'mv', 'mv')
transfer_array(m_open, m_closed, 'autolabels', 'autolabels')
# m_final = pointthreshold(m_closed, 'mv', 0, 0)
# writevtk(m_closed, args.meshfile_open[:-4]+"-zone-m_closed.vtk")
m_final = pointthreshold(m_closed, 'zone', .001, 5)

path_tmp = os.path.dirname(args.meshfile_open)+"/m_final"
writevtk(m_final, path_tmp+".vtk")
png_writestl(path_tmp+".vtk", path_tmp +".stl")
surface_reconst(path_tmp+".stl", path_tmp+".stl", depth=7)
writevtk(png_readstl(path_tmp+".stl"), path_tmp+".vtk")

m_final2 = readvtk(path_tmp+".vtk")

transfer_all_scalar_arrays(m_final, m_final2)
writevtk(m_final2, path_tmp+"2.vtk")

# png_writestl(args.meshfile_open, args.meshfile_open[:-4]+".stl")
# surface_reconst(args.meshfile_open[:-4]+".stl", args.meshfile_closed[:-4]+".stl", depth=7)
# writevtk(png_readstl(args.meshfile_closed[:-4]+".stl"), args.meshfile_closed)




# m_final.GetPointData().RemoveArray('mv')
writevtk(m_final, args.meshfile_closed, 'ascii')


"""
png_mesh_GetArray(m_final).shape
png_visualize1(m_final)
png_visualize1(m_closed)
png_mesh_GetArray(m_closed)
png_cell_GetArray(m_final).shape

import pandas as pd
writevtk(m_closed, 'E:/data/map/tmp/flat-3218/tmp2.vtk')
np.where(png_mesh_GetArray2(m_closed, 'zone')<1)
np.where(png_mesh_GetArray2(m_closed, 'mv')>0.5)
pd.DataFrame()
"""

print('PV and LAA holes have been closed and marked with scalar array <hole> = 1')
