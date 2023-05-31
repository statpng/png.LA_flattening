"""
path = "C:/Users/statpng/Desktop/1.Mesh3D/LA_flattening/data/"
filename = "mesh"

meshfile = path+filename+".vtk"
meshfile_open = path+filename+"_crinkle_clipped.vtk" 
meshfile_open_no_mitral = path+filename+"_clipped_mitral.vtk" 
meshfile_closed = path+filename+"_clipped_c.vtk"

import os
os.getcwd()
os.chdir("C://Users//statpng//Desktop//1.Mesh3D//LA_flattening")


from aux_functions import *
from clip_aux_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

"""


import vtk
from png_functions import *
from aux_functions import *
from clip_aux_functions import *
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


###     Input/Output    ###
def png_readvtk(filename, type_vtk="polydata"):
    """Read VTK file"""
    if type_vtk == "unstructured":
        reader = vtk.vtkUnstructuredGridReader()
    elif type_vtk == "polydata":
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def png_readvtp(filename):
    """Read VTP file"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def png_writevtk(surface, filename, type="polydata", encoding='ascii'):
    """Write binary or ascii VTK file"""
    if type == "unstructured":
        writer = vtk.vtkUnstructuredGridWriter()
    elif type == "polydata":
        writer = vtk.vtkPolyDataWriter()

    writer.SetInputData(surface)
    writer.SetFileName(filename)
    if encoding == 'ascii':
        writer.SetFileTypeToASCII()
    elif encoding == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()

def png_writevtp(surface, filename):
    """Write VTP file"""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
#    writer.SetDataModeToBinary()
    writer.Write()





def visualise_default(surface, ref, case, arrayname, mini, maxi, type="polydata"):
    """Visualise surface with a default parameters"""
    #Create a lookup table to map cell data to colors
    # print "Colormap from ", mini, "to", maxi
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(255)
    lut.SetValueRange(0, 255)

    # qualitative data from colorbrewer  --> matching qualitative colormap of Paraview
    lut.SetTableValue(0, 0, 0, 0, 1)  #Black
    lut.SetTableValue(mini, 1, 1, 1, 1)   #white
    lut.SetTableValue(mini+1, 77/255.,175/255., 74/255., 1)   # green
    lut.SetTableValue(maxi-3, 152/255.,78/255.,163/255., 1)  # purple
    lut.SetTableValue(maxi-2, 255/255.,127/255., 0., 1)  # orange
    lut.SetTableValue(maxi-1, 55/255., 126/255., 184/255., 1)  # blue
    lut.SetTableValue(maxi, 166/255., 86/255., 40/255., 1)  # brown
    lut.Build()

    # create a text actor
    txt = vtk.vtkTextActor()
    txt.SetInput(case)
    txtprop=txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(18)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(20, 30)

    # create a rendering window, renderer, and renderwindowinteractor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    # for GIMIAS interaction style
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(renWin)

    # surface mapper and actor
    if type == "polydata":
        surfacemapper = vtk.vtkPolyDataMapper()
    elif type == "unstructured":
        surfacemapper = vtk.vtkUnstructuredDataMapper()
        
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        surfacemapper.SetInputData(surface)
    else:
        surfacemapper.SetInput(surface)
        
    surfacemapper.SetScalarModeToUsePointFieldData()
    surfacemapper.SelectColorArray(arrayname)
    surfacemapper.SetLookupTable(lut)
    surfacemapper.SetScalarRange(0,255)
    surfaceactor = vtk.vtkActor()
    # surfaceactor.GetProperty().SetOpacity(0)
    # surfaceactor.GetProperty().SetColor(1, 1, 1)
    surfaceactor.SetMapper(surfacemapper)

    # refsurface mapper and actor
    refmapper = vtk.vtkPolyDataMapper()
    if vtk.vtkVersion.GetVTKMajorVersion() > 5:
        refmapper.SetInputData(ref)
    else:
        refmapper.SetInput(ref)
    refmapper.SetScalarModeToUsePointFieldData()
    refmapper.SelectColorArray(arrayname)
    refmapper.SetLookupTable(lut)
    refmapper.SetScalarRange(0,255)
    refactor = vtk.vtkActor()
    refactor.GetProperty().SetOpacity(0.5)
    # refactor.GetProperty().SetColor(1, 1, 1)
    refactor.SetMapper(refmapper)

    # assign actors to the renderer
    ren.AddActor(refactor)
    ren.AddActor(surfaceactor)
    ren.AddActor(txt)

    # set the background and size; zoom in; and render
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1280, 960)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1)

    # before
    # print("before", ren.GetActiveCamera().GetViewUp())

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()

    outcam = ren.GetActiveCamera()
    # print("after", outcam.GetViewUp())


def png_visualize1(mesh):
    visualise_default(mesh, mesh, 'STD mesh', 'autolabels', 36, 79)
    # visualise_default(mesh, mesh, 'STD mesh', 'autolabels', 10, 39)

def png_visualize2(mesh):
    # visualise_default(mesh, mesh, 'STD mesh', 'autolabels', 36, 79)
    visualise_default(mesh, mesh, 'STD mesh', 'autolabels', 10, 39)















import os 
import pyvista as pv 

# def vtk2stl(path, fname):
#     vtk_f = pv.read(os.path.join(path, fname))
#     pv.save_meshio(path+fname[:-4] + '.stl', vtk_f)
#     print(fname + ' is changed to STL !')

# def stl2vtk(path, fname): 
#     stl_f = pv.read(os.path.join(path, fname))
#     pv.save_meshio(path+fname[:-4] + '.vtk', stl_f)
#     print(fname + ' is changed to VTK !')

def stl2vtk(From, To): 
    stl_f = pv.read(os.path.join(From))
    pv.save_meshio(To, stl_f)
    print('changed to VTK !')
    
def vtk2stl(From, To): 
    stl_f = pv.read(os.path.join(From))
    pv.save_meshio(To, stl_f)
    print('changed to stl !')
    
# vtk2stl(path, filename+"_crinkle_clipped.vtk")
# stl2vtk(path, filename+"_clipped_c.stl")



def png_writestl(path, filename, type="ascii"):
    """
    path = args.meshfile_open
    filename = input_mesh_file_stl
    """
    surface = png_readvtk(path)
    
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(surface)
    writer.SetFileName(filename)
    if type == 'ascii':
        writer.SetFileTypeToASCII()
    elif type == 'binary':
        writer.SetFileTypeToBinary()
    writer.Write()




def png_readstl(filename):
    """
    filename = input_stl_file
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()



def convert_stl_to_structured_vtk(input_stl_file, output_vtk_file, resolution):
    # Load the STL mesh
    mesh = pv.read(input_stl_file)

    # Compute the axis-aligned bounding box
    aabb_min = mesh.bounds[::2]
    aabb_max = mesh.bounds[1::2]

    # Create a uniform grid for resampling
    grid = pv.UniformGrid()
    grid.dimensions = np.array([resolution, resolution, resolution])
    grid.origin = aabb_min
    grid.spacing = (np.array(aabb_max) - np.array(aabb_min)) / (np.array([resolution, resolution, resolution]) - 1)

    # Resample the mesh onto the grid
    resampled_mesh = mesh.sample(grid)

    # Save the structured VTK file
    resampled_mesh.save(output_vtk_file)






def fillholes(surface, holesize=1000):
    """Fill holes in surface. Use holesize to specify the max 'radius' of the
    holes to be filled."""
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(surface)
    filler.SetHoleSize(holesize)
    filler.Update()
    return filler.GetOutput()


def surface_reconst(input_mesh_file, output_mesh_file, method="generate_surface_reconstruction_screened_poisson", depth=8):
    import pymeshlab as ml
    ms = ml.MeshSet()
    # ms.add_mesh(input_mesh_file)
    ms.load_new_mesh(input_mesh_file)
    
    ms.generate_surface_reconstruction_screened_poisson(depth=depth) 
    ms.meshing_repair_non_manifold_edges(method=0)
    
    # CLOSE_HOLES_LIMIT = 250
    # ms.meshing_close_holes(maxholesize=CLOSE_HOLES_LIMIT)
    
    # ms.apply_filter("generate_surface_reconstruction_screened_poisson")
    # ml.print_filter_list()
    ms.save_current_mesh(output_mesh_file)




def png_print_scalar_values(polydata):
    point_data = polydata.GetPointData()
    cell_data = polydata.GetCellData()

    num_point_arrays = point_data.GetNumberOfArrays()
    num_cell_arrays = cell_data.GetNumberOfArrays()

    print("Point Data Scalar Arrays:")
    for i in range(num_point_arrays):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)
        print(f"  Array {i}: {array_name} - {array}")

    print("Cell Data Scalar Arrays:")
    for i in range(num_cell_arrays):
        array_name = cell_data.GetArrayName(i)
        array = cell_data.GetArray(i)
        print(f"  Array {i}: {array_name} - {array}")

# Example usage:
# polydata = ... (Load or create your vtkPolyData object)
# print_scalar_values(polydata)





def png_mesh_GetArray2(mesh, name):
    import numpy as np
    
    data = mesh.GetPointData()
    out = []
    for i in range(data.GetNumberOfArrays()):
        ArrayName = data.GetArray(i).GetName()
        print( ArrayName )
    
    
    scalar_array = data.GetArray(name)
    numpy_array = vtk_to_numpy(scalar_array)
        
    
    return(numpy_array)



def png_mesh_GetArray(mesh):
    import numpy as np
    
    data = mesh.GetPointData()
    for i in range(data.GetNumberOfArrays()):
        ArrayName = data.GetArray(i).GetName()
        scalar_array = data.GetArray(ArrayName)
        numpy_array = vtk_to_numpy(scalar_array)
        print( ArrayName )
        print( np.histogram(numpy_array) )
    return(numpy_array)



def png_writePointArray(mesh, filename):
    """
    mesh = m_final
    filename = m_out[:-4]+".csv"
    """
    
    import numpy as np
    import pandas as pd
    
    
    data = mesh.GetPointData()
    
    array_dict = {}
    for i in range(data.GetNumberOfArrays()):
        ArrayName = data.GetArray(i).GetName()
        scalar_array = data.GetArray(ArrayName)
        numpy_array = vtk_to_numpy(scalar_array)[:,np.newaxis]
        array_dict[f'{ArrayName}'] = numpy_array
        
        numpy_array.shape
        print( ArrayName )
    
    
    df_list = [pd.DataFrame(array_dict[key], columns=[f'{key}' for i in range(array.shape[1])]) for key, array in array_dict.items()]

    result_df = pd.concat(df_list, axis=1)
    
    base_columns = ["x","y","z","region","mv","hole","pv","autolabels","zone"]
    
    [base_columns.append(x) for x in result_df.columns.to_list() if not x in base_columns ]
    
    result_df[base_columns].to_csv(filename, index=False)
    
    return(result_df )







def png_writeCoord(surface, filename, type="nodes"):
    """
    mesh = m_final
    filename = m_out[:-4]+".csv"
    """
    
    import numpy as np
    import pandas as pd
    
    
    if type == "edges":
        df_edges = pd.DataFrame( ExtractVTKTriFaces(surface) )
        df_edges.to_csv(filename, index=False)
    elif type == "nodes":
        df_nodes = pd.DataFrame( png_mesh_GetCoord(surface) )
        df_nodes.to_csv(filename, index=False)
        
    return 1




def png_read_and_write_coord(path_nodes, path_edges, path_out):
    """
    path_nodes = "E:/download/reconst_test - nodes.csv"
    path_edges = "E:/download/reconst_test - edges.csv"
    path_out = "E:/download/reconst_test.vtk"
    
    path_nodes = "E:/download/reconst_test_org - nodes.csv"
    path_edges = "E:/download/reconst_test_org - edges.csv"
    path_out = "E:/download/reconst_test_org.vtk"
    
    png_read_coord(path_nodes, path_edges, path_out)
    """
    import vtk
    import pandas as pd

    nodes = pd.read_csv(path_nodes)
    edges = pd.read_csv(path_edges)
    values = nodes.drop(['x','y','z'], axis=1)
    
    fnames = values.columns
    
    points = vtk.vtkPoints()
    for index, row in nodes.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    
    for fname in fnames:
        values = vtk.vtkFloatArray()
        values.SetName(fname)
        
        for index, row in nodes.iterrows():
            values.InsertNextValue(row[fname])
            
        polydata.GetPointData().AddArray(values)
    
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x']-1)
        triangle.GetPointIds().SetId(1, row['y']-1)
        triangle.GetPointIds().SetId(2, row['z']-1)
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    # vtkPolyData를 파일로 저장
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path_out)
    writer.SetInputData(polydata)
    writer.Write()
    
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName("output.vtp")
    # writer.SetInputData(polydata)
    # writer.Write()
    
    return polydata



    



def png_cell_GetArray(mesh):
    import numpy as np
    
    data = mesh.GetCellData()
    for i in range(data.GetNumberOfArrays()):
        ArrayName = data.GetArray(i).GetName()
        scalar_array = data.GetArray(ArrayName)
        numpy_array = vtk_to_numpy(scalar_array)
        print( ArrayName )
        print( np.histogram(numpy_array) )
    return(numpy_array)





def png_cell_GetArray2(mesh, arrname):
    import numpy as np
    
    data = mesh.GetCellData()
    scalar_array = data.GetArray(arrname)
    numpy_array = vtk_to_numpy(scalar_array)
    
    print( arrname )
    print( np.histogram(numpy_array) )

    return(numpy_array)





def png_mesh_GetCoord(mesh):
    import numpy as np
    
    data = mesh.GetPoints()
    return vtk_to_numpy( data.GetData() )
    

"""
mesh_org = readvtk("C:/Users/statpng/Desktop/1.Mesh3D/LA_flattening - org/data_fig1/output/mesh_clipped_c.vtk")
png_mesh_GetArray(mesh_org)

png_mesh_GetCoord(m_open)
png_mesh_GetCoord(m_closed)
"""










def R_gsub(string, pattern="\\", replacement="/"):
    # string = r"E:\data\map\Voltage\conv.cuv"
    output = string.replace(pattern, replacement)

    return output


def png_plt2polydata(path, path_out, fname):
    # path = "E:/data/map/Voltage/conv.plt/5645-VoltMap.plt"
        
    import re
    
    file = open(path, 'r')
    lines = file.readlines()[1:]
    header = lines[0]
    line = lines[1:]
    filename = re.findall(r'[^\\/]+(?=\.plt$)', path)[0]
    
    
    change_index = int( re.findall(r"\d+", header)[0] )
    nodes = line[:change_index]
    edges = line[change_index:]
    
    # row=nodes[0]
    import pandas as pd
    nodes_split = [list(map(float, row.strip().split())) for row in nodes]
    edges_split = [list(map(int, row.strip().split())) for row in edges]
    
    nodes_df = pd.DataFrame(nodes_split , columns=['x', 'y', 'z', "value"])
    edges_df = pd.DataFrame(edges_split , columns=['x', 'y', 'z'])

    nodes_df.head()
    edges_df.head()
    
    
    points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName(fname)
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue(row['value'])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)
    
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x']-1)
        triangle.GetPointIds().SetId(1, row['y']-1)
        triangle.GetPointIds().SetId(2, row['z']-1)
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    # vtkPolyData를 파일로 저장
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path_out)
    writer.SetInputData(polydata)
    writer.Write()
    
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName("output.vtp")
    # writer.SetInputData(polydata)
    # writer.Write()
    
    return polydata






def png_plt2node(path):
    # path = "E:/data/map/Voltage/conv.plt/5645-VoltMap.plt"
        # path_out
    import re
    
    file = open(path, 'r')
    lines = file.readlines()[1:]
    header = lines[0]
    line = lines[1:]
    filename = re.findall(r'[^\\/]+(?=\.plt$)', path)[0]
    
    
    change_index = int( re.findall(r"\d+", header)[0] )
    nodes = line[:change_index]
    import pandas as pd
    nodes_split = [list(map(float, row.strip().split())) for row in nodes]
    nodes_df = pd.DataFrame(nodes_split , columns=['x', 'y', 'z', "value"])
    nodes_df.head()
    
    return nodes_df






def decrease_resolution(mesh, prop):
    import vtk
    
    # 원본 메시: polydata (이미 생성되었다고 가정)
    
    # Quadric decimation 필터 생성
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(polydata)
    
    # 타겟 해상도 설정 (0.1은 원래 메시의 10%를 의미)
    decimator.SetTargetReduction(prop)
    decimator.Update()
    
    # 결과 메시
    decimated_polydata = decimator.GetOutput()
    
    return decimated_polydata






def png_readcuvia(path, level=2):
    # path = "E:/data/map/Voltage/raw.cuv/5645-outputXML.cuv"
    # outpath = "E:/data/map/Voltage/conv.cuv"
    
    
    import re
    import pandas as pd
    
    filename = re.findall(r'[^\\/]+(?=\.cuv$)', path)[0]
    
    skip_rows = 4
    start = 0
    
    
    for i in range(level):
        if i > 0 :
            start = edge_end + 1
        
        file = open(path, 'r')
        lines = file.readlines()[(skip_rows+start):(skip_rows+start+1)]
        
        header = lines[0]
        # line = lines[1:]
        
        row = list(map(int, re.findall(r"\d+", header) ))
        node_start = start
        node_end = start + row [0]
        edge_start = start + row [0] + 1
        edge_end = start + row [0] + row[1]
    
    
    file = open(path, 'r')
    lines = file.readlines()[(skip_rows+1):]
    
    
    nodes = lines[node_start:node_end]
    edges = lines[node_end:edge_end]
    
    
    nodes_split = [list(map(float, row.strip().split())) for row in nodes]
    edges_split = [list(map(int, row.strip().split()[:3])) for row in edges]
    
    nodes_df = pd.DataFrame(nodes_split , columns=['x', 'y', 'z'])
    edges_df = pd.DataFrame(edges_split , columns=['x', 'y', 'z'])


    return nodes_df, edges_df
    



def png_cuvia2stl(path_cuv, path_cuv2stl, level=1):
    # path_cuv = "E:/data/map/Voltage/raw.cuv/5645-outputXML.cuv"
    # path_cuv2stl = "E:/data/map/Voltage/conv.cuv"
    
    
    import re
    import pandas as pd
    
    filename = re.findall(r'[^\\/]+(?=\.cuv$)', path_cuv)[0]
    
    skip_rows = 4
    start = 0
    
    
    for i in range(level):
        if i > 0 :
            start = edge_end + 1
        
        file = open(path_cuv, 'r')
        lines = file.readlines()[(skip_rows+start):(skip_rows+start+1)]
        
        header = lines[0]
        # line = lines[1:]
        
        row = list(map(int, re.findall(r"\d+", header) ))
        node_start = start
        node_end = start + row [0]
        edge_start = start + row [0] + 1
        edge_end = start + row [0] + row[1]
    
    
    file = open(path_cuv, 'r')
    lines = file.readlines()[(skip_rows+1):]
    
    
    nodes = lines[node_start:node_end]
    edges = lines[node_end:edge_end]
    
    nodes[:5]
    edges[:5]
    
    nodes_split = [list(map(float, row.strip().split())) for row in nodes]
    edges_split = [list(map(int, row.strip().split()[:4])) for row in edges]
    
    nodes_df = pd.DataFrame(nodes_split , columns=['x', 'y', 'z'])
    edges_df = pd.DataFrame(edges_split , columns=['x', 'y', 'z', 'zone'])


    nodes_df.head()
    edges_df.head()
    
    
    points = vtk.vtkPoints()
    # values = vtk.vtkFloatArray()
    # values.SetName("Values")
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        # values.InsertNextValue(row['value'])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    # polydata.GetPointData().AddArray(values)
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    
    polydata.SetPolys(cells)
    
    # vtkPolyData를 파일로 저장
    # writer = vtk.vtkPolyDataWriter()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path_cuv2stl)
    writer.SetInputData(polydata)
    writer.Write()
    
    
    



def png_cuvia2polydata(path, outpath, level=1):
    # path_cuv, path_cuv2vtk, level=1
    # path = "E:/data/map/tmp/3218-Cuvia.cuv"
    # outpath = "E:/data/map/tmp/3218-Cuvia.vtk"
    
    
    import re
    import pandas as pd
    
    filename = re.findall(r'[^\\/]+(?=\.cuv$)', path)[0]
    
    skip_rows = 4
    start = 0
    
    
    for i in range(level):
        if i > 0 :
            start = edge_end + 1
        
        file = open(path, 'r')
        lines = file.readlines()[(skip_rows+start):(skip_rows+start+1)]
        
        header = lines[0]
        # line = lines[1:]
        
        row = list(map(int, re.findall(r"\d+", header) ))
        node_start = start
        node_end = start + row [0]
        edge_start = start + row [0] + 1
        edge_end = start + row [0] + row[1]
    
    
    file = open(path, 'r')
    lines = file.readlines()[(skip_rows+1):]
    
    
    nodes = lines[node_start:node_end]
    edges = lines[node_end:edge_end]
    
    nodes[:5]
    edges[:5]
    
    nodes_split = [list(map(float, row.strip().split())) for row in nodes]
    edges_split = [list(map(int, row.strip().split()[:4])) for row in edges]
    
    nodes_df = pd.DataFrame(nodes_split , columns=['x', 'y', 'z'])
    edges_df = pd.DataFrame(edges_split , columns=['x', 'y', 'z', 'zone'])


    nodes_df.head()
    edges_df.head()
    
    
    points = vtk.vtkPoints()
    values_x = vtk.vtkFloatArray()
    values_y = vtk.vtkFloatArray()
    values_z = vtk.vtkFloatArray()
    values_x.SetName("x")
    values_y.SetName("y")
    values_z.SetName("z")
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values_x.InsertNextValue(row['x'])
        values_y.InsertNextValue(row['y'])
        values_z.InsertNextValue(row['z'])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values_x)
    polydata.GetPointData().AddArray(values_y)
    polydata.GetPointData().AddArray(values_z)
    
    
    
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    zone_vtkarray = numpy_to_vtk(np.array(edges_df['zone']))
    zone_vtkarray.SetName('zone')
    polydata.GetCellData().AddArray(zone_vtkarray)
    
    # polydata=cellthreshold(polydata, "zone", 1, 2)
    
    
    # vtkPolyData를 파일로 저장
    writer = vtk.vtkPolyDataWriter()
    # writer = vtk.vtkSTLWriter()
    writer.SetFileName(outpath)
    writer.SetInputData(polydata)
    writer.Write()
    
    
    # polydata = readvtk(outpath)
    # polydata = convert_cell_data_to_point_data( polydata )
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName(outpath)
    # writer.SetInputData(polydata)
    # writer.Write()
    
    """
    png_visualize1(polydata)
    """
    
    
    
    


def subsample_points_poisson(inputMesh, radius=4.5):
    """
        Return sub-sampled points as numpy array.
        The radius might need to be tuned as per the requirements.
    """
    import vtk
    from vtk.util import numpy_support

    f = vtk.vtkPoissonDiskSampler()
    f.SetInputData(inputMesh)
    f.SetRadius(radius)
    f.Update()

    sampled_points = f.GetOutput()
    points = sampled_points.GetPoints()
    pointdata = points.GetData()
    out = numpy_support.vtk_to_numpy(pointdata)
    
    
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(sampled_points)
    delaunay.Update()

    # vtkUnstructuredGrid에서 vtkPolyData로 변환
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(delaunay.GetOutput())
    geometry_filter.Update()
    triangular_mesh = geometry_filter.GetOutput()
    
    return triangular_mesh

    # 결과를 VTK 파일로 저장
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName("triangular_mesh.vtk")
    # writer.SetInputData(triangular_mesh)
    # writer.Write()


    # return out

def png_pointset2polydata(point_set, vertex_set):
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(point_set.GetPoints())
    
    vertices = vtk.vtkCellArray()

    # 각 포인트를 정점으로 사용하는 셀을 추가
    for i in range(point_set.GetNumberOfPoints()):
        triangle = vtk.vtkTriangle()
        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)  # 정점의 인덱스를 설정
        vertices.InsertNextCell(vertex)
 
    # 생성된 셀 정보를 vtkPolyData에 설정
    poly_data.SetVerts(vertices)
     
    return poly_data


# Convert from numpy to vtk points
def numpy_to_vtk_points(input_points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(input_points, deep=True))
    
    return vtk_points

# Convert from numpy to vtkPolyData
def numpy_to_vtk_polydata(input_points):
    vtk_polydata = vtk.vtkPolyData()
    vtk_points = numpy_to_vtk_points(input_points)
    vtk_polydata.SetPoints(vtk_points)
    
    return vtk_polydata


def png_readstl(filename):
    """Read STL file"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()
















def png_match_flat2vtk(path_vtk, path_flat, path_out, type_vtk="polydata"):
    """
    path_vtk=path_vtk;  path_flat=final_flat;  path_out=path_out;  type_vtk="polydata"
    """
    import re
    import pandas as pd
    import numpy as np
    
    
    mesh_flat = png_readvtk(path_flat, type_vtk="polydata")
    
    mesh_flat = convert_cell_data_to_point_data( mesh_flat )
    
    cuv_x = png_mesh_GetArray2(mesh_flat, "x")[:,np.newaxis]
    cuv_y = png_mesh_GetArray2(mesh_flat, "y")[:,np.newaxis]
    cuv_z = png_mesh_GetArray2(mesh_flat, "z")[:,np.newaxis]
    cuv_nodes = np.concatenate((cuv_x, cuv_y, cuv_z), axis=1)
    
    
    
    
    fname = path_vtk.split("-")[-1].split(".")[0]
    # re.search("-([a-zA-Z]+).vtk", path_vtk).group(0)
    mesh = png_readvtk(path_vtk, type_vtk=type_vtk)
    
    mesh_value = png_mesh_GetArray(mesh)
    mesh_coord = png_mesh_GetCoord(mesh)
    
    
    
    var_ratio = cuv_nodes[:,range(2)].std(axis=0) / mesh_coord[:,range(2)].std(axis=0)
    var_ratio_argmin = np.argmin( np.abs(np.max(var_ratio) - np.array([0.1,1,10])) )
    var_ratio_unique = [0.1,0,10][var_ratio_argmin]
    if var_ratio_unique > 0:
        # cuv_nodes /= var_ratio
        mesh_coord *= var_ratio_unique
    
    
    matching_index_set=[]
    matching_norm=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        """
        row = cuv_nodes[1]
        """
        
        matching_rows = np.linalg.norm(mesh_coord - row, axis=1)
        np.sum(matching_rows < 1)
        
        if(np.sum(matching_rows < 1) > 0):
            matching_index = int(np.argmin(matching_rows))
            matching_index_set.append( matching_index )
            
            matching_norm.append( matching_rows[matching_index] )
        
    cuv_values = mesh_value[matching_index_set]
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    
    
    print( len(cuv_nodes) )
    print( len(cuv_values) )
    
    
    cuv_nodes_flat = png_mesh_GetCoord(mesh_flat)
    cuv_edges_flat = ExtractVTKTriFaces(mesh_flat)
    
    
    nodes_df = pd.DataFrame( np.hstack((cuv_nodes_flat, cuv_values_2d)), columns=["x","y","z",fname] )
    edges_df = pd.DataFrame( cuv_edges_flat, columns=["x","y","z"] )
       
    
    
    
    
    
    
    polydata = vtk.vtkPolyData()
    
    # Point
    points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName(fname)
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue(row[fname])
        
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)

    
    
    for x in ["x", "y", "z", "region", "mv", "hole", "pv", "autolabels", "zone"]:
        vtkarray_x = numpy_to_vtk( png_mesh_GetArray2(mesh_flat, x) )
        vtkarray_x.SetName(x)
        polydata.GetPointData().AddArray(vtkarray_x)
    
    
    
    
    # Cell
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    
    
    # png_mesh_GetArray(polydata, "DF")
        
    png_writevtk(polydata, filename=path_out, type="polydata")
    
    

    
    
    
    
    
    


def png_match_region(mesh_flat, path_region):
    """
    mesh_flat = combined_polydata
    path_region = path_auto+"/df_region_"+filename+".csv"
    """
    import re
    import pandas as pd
    import numpy as np
    
    
    cuv_x = png_mesh_GetArray2(mesh_flat, "x")[:,np.newaxis]
    cuv_y = png_mesh_GetArray2(mesh_flat, "y")[:,np.newaxis]
    cuv_z = png_mesh_GetArray2(mesh_flat, "z")[:,np.newaxis]
    cuv_nodes = np.concatenate((cuv_x, cuv_y, cuv_z), axis=1)
    
    
    df_region = pd.read_csv(path_region)
    df_region_coord = np.array( df_region[["x","y","z"]] )
    df_region_value = df_region["region"].to_numpy()
    
    
    
    var_ratio = cuv_nodes[:,range(2)].std(axis=0) / df_region_coord[:,range(2)].std(axis=0)
    var_ratio_argmin = np.argmin( np.abs(np.max(var_ratio) - np.array([0.1,1,10])) )
    var_ratio_unique = [0.1,0,10][var_ratio_argmin]
    if var_ratio_unique > 0:
        # cuv_nodes /= var_ratio
        mesh_coord *= var_ratio_unique
    
    
    matching_index_set=[]
    matching_norm=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        """
        row = cuv_nodes[0]
        """
        
        matching_rows = np.linalg.norm(df_region_coord - row, axis=1)
        np.sum(matching_rows < 1)
        
        nonexact = 0
        if np.sum(matching_rows < 1) > 0:
            nonexact += int( int(np.min(matching_rows)) > 0 )
            matching_index = int(np.argmin(matching_rows))
            matching_index_set.append( matching_index )
            
            matching_norm.append( matching_rows[matching_index] )
        
    cuv_values = df_region_value[matching_index_set]
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    
    
    print( len(cuv_nodes) )
    print( len(cuv_values) )
    print( "# of non-exact cases = " + str(nonexact) )

    
    import pandas as pd
    cuv_nodes_flat = png_mesh_GetCoord(mesh_flat)
    cuv_edges_flat = ExtractVTKTriFaces(mesh_flat)
    
    
    nodes_df = pd.DataFrame( np.hstack((cuv_nodes_flat, cuv_values_2d)), columns=["x","y","z","auto_region"] )
    edges_df = pd.DataFrame( cuv_edges_flat, columns=["x","y","z"] )
    nodes_df["auto_region"] = [ int(re.findall(r"\d+", x)[0]) for x in nodes_df["auto_region"] ]
    
    
    
    polydata = vtk.vtkPolyData()
    
    # Point
    points = vtk.vtkPoints()
    values = vtk.vtkIntArray()
    values.SetName("auto_region")
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue( row["auto_region"] )
    
    
        
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)

    
    for x in [ mesh_flat.GetPointData().GetArrayName(idx) for idx in range(mesh_flat.GetPointData().GetNumberOfArrays()) ]:
        vtkarray_x = numpy_to_vtk( png_mesh_GetArray2(mesh_flat, x) )
        vtkarray_x.SetName(x)
        polydata.GetPointData().AddArray(vtkarray_x)
    
    
    
    
    # Cell
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    
    return polydata
    
    
    
    
    
    
    
    
    
def png_match_vtk2cuvvtk(path_vtk, path_cuvcutvtk, path_out, type_vtk="polydata"):
    """
    path_vtk=path_vtk;  path_cuvcutvtk=final_flat;  path_out=path_out;  type_vtk="polydata"
    """
    import re
    
    mesh_cuvvtk = png_readvtk(path_cuvcutvtk, type_vtk="unstructured")
    
    cuv_nodes = png_mesh_GetCoord(mesh_cuvvtk)
    cuv_edges = ExtractVTKTriFaces(mesh_cuvvtk)
    cuv_zone = png_cell_GetArray2(mesh_cuvvtk, "zone")
    # coord_x = png_cell_GetArray2(mesh_cuvvtk, "x")
    # coord_y = png_cell_GetArray2(mesh_cuvvtk, "y")
    # coord_z = png_cell_GetArray2(mesh_cuvvtk, "z")
    
    
    fname = path_vtk.split("-")[-1].split(".")[0]
    # re.search("-([a-zA-Z]+).vtk", path_vtk).group(0)
    mesh = png_readvtk(path_vtk, type_vtk=type_vtk)
    
    mesh_value = png_mesh_GetArray(mesh)
    mesh_coord = png_mesh_GetCoord(mesh)
    

    var_ratio = np.unique( np.round( cuv_nodes.std(axis=0) / mesh_coord.std(axis=0), -1 ) )
    if var_ratio > 0:
        # cuv_nodes /= var_ratio
        mesh_coord *= var_ratio
    
    
    matching_index_set=[]
    cuv_values=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        matching_rows = np.all(np.abs(mesh_coord - row) < 1e-3, axis=1)
        if(np.sum(matching_rows) > 0):
            matching_index = int(np.where(matching_rows)[0])
            matching_value = mesh_value[matching_index]
        else:
            matching_index = np.nan
            matching_value = np.nan
        
        matching_index_set.append( matching_index )
        cuv_values.append( matching_value )
        
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    print( len(cuv_nodes) )
    print( len(cuv_values) )

    
    import pandas as pd
        
    nodes_df = pd.DataFrame( np.hstack((cuv_nodes, cuv_values_2d)), columns=["x","y","z",fname] )
    edges_df = pd.DataFrame( cuv_edges, columns=["x","y","z"] )
    zone_df = pd.DataFrame( cuv_zone )
    
    
    # Point
    points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName(fname)
    values_x = vtk.vtkFloatArray()
    values_y = vtk.vtkFloatArray()
    values_z = vtk.vtkFloatArray()
    values_x.SetName("x")
    values_y.SetName("y")
    values_z.SetName("z")
    
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue(row[fname])
        values_x.InsertNextValue(row['x'])
        values_y.InsertNextValue(row['y'])
        values_z.InsertNextValue(row['z'])
        
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)
    polydata.GetPointData().AddArray(values_x)
    polydata.GetPointData().AddArray(values_y)
    polydata.GetPointData().AddArray(values_z)
    
    
    # Cell
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    zone_vtkarray = numpy_to_vtk(np.array(zone_df))
    zone_vtkarray.SetName('zone')
    polydata.GetCellData().AddArray(zone_vtkarray)
    
    
    
    # png_mesh_GetArray(polydata, "DF")
        
    png_writevtk(polydata, filename=path_out, type="polydata")
    
    
    



def png_unstructured2polydata(path):
    import pandas as pd
    import numpy as np
    
    mesh = png_readvtk(path, type_vtk="unstructured")
    
    nodes_df = pd.DataFrame( png_mesh_GetCoord(mesh), columns=["x","y","z"] )
    edges_df = pd.DataFrame( ExtractVTKTriFaces(mesh), columns=["x","y","z"] )
    zone_df = png_cell_GetArray2(mesh, "zone")
    
    # Point
    points = vtk.vtkPoints()
    
    values_x = vtk.vtkFloatArray()
    values_y = vtk.vtkFloatArray()
    values_z = vtk.vtkFloatArray()
    values_x.SetName("x")
    values_y.SetName("y")
    values_z.SetName("z")
    
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values_x.InsertNextValue(row['x'])
        values_y.InsertNextValue(row['y'])
        values_z.InsertNextValue(row['z'])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values_x)
    polydata.GetPointData().AddArray(values_y)
    polydata.GetPointData().AddArray(values_z)
    
    
    # Cell
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    zone_vtkarray = numpy_to_vtk(np.array(zone_df))
    zone_vtkarray.SetName('zone')
    polydata.GetCellData().AddArray(zone_vtkarray)
    
    save_path = path[:-4]+"_poly.vtk"
    png_writevtk(polydata, filename=save_path, type="polydata")
    
    # return polydata
    
    
    



def png_match_stl2vtk(path_cuvcutstl, path_cuv2vtk, path_out, type_vtk="polydata"):
    
    mesh_cut = png_readstl(path_cuvcutstl)
    mesh_vtk = png_readvtk(path_cuv2vtk)
    
    cut_nodes = png_mesh_GetCoord(mesh_cut)
    cut_edges = ExtractVTKTriFaces(mesh_cut)
    
    vtk_nodes = png_mesh_GetCoord(mesh_vtk)
    vtk_edges = ExtractVTKTriFaces(mesh_vtk)
    vtk_zone = png_cell_GetArray(mesh_vtk)
    
    
    
    cut_nodes[0]
    cut_edges
    vtk_nodes[3788]
    vtk_edges
    
    enum = enumerate(cut_nodes)
    index, row = next(enum)
    
    matching_rows = np.all(np.abs(vtk_nodes - row) < 1e-3, axis=1)
    int(np.where(matching_rows)[0])
    
    if(np.sum(matching_rows) > 0):
        matching_index = int(np.where(matching_rows)[0])
        matching_value = vtk_zone[matching_index]
    else:
        matching_index = np.nan
        matching_value = np.nan
    
    
    row
    
    
    matching_index_set=[]
    cuv_values=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        matching_rows = np.all(np.abs(mesh_coord - row) < 1e-3, axis=1)
        if(np.sum(matching_rows) > 0):
            matching_index = int(np.where(matching_rows)[0])
            matching_value = mesh_value[matching_index]
        else:
            matching_index = np.nan
            matching_value = np.nan
        
        matching_index_set.append( matching_index )
        cuv_values.append( matching_value )
        
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    
    
    mesh_vtk

    
    


def png_match_vtk2cuvstl(path_vtk, path_cuvcutstl, path_cuv2vtk, path_out, type_vtk="polydata"):

    import re
    
    mesh_cuvcutstl = png_readstl(path_cuvcutstl)
    mesh_cuvvtk = png_readvtk(path_cuv2vtk)
    
    cuv_nodes = png_mesh_GetCoord(mesh_cuvcutstl)
    cuv_edges = ExtractVTKTriFaces(mesh_cuvcutstl)

    fname = path_vtk.split("-")[-1].split(".")[0]
    # re.search("-([a-zA-Z]+).vtk", path_vtk).group(0)
    mesh = png_readvtk(path_vtk, type_vtk=type_vtk)
    
    mesh_value = png_mesh_GetArray(mesh)
    mesh_coord = png_mesh_GetCoord(mesh)
    

    var_ratio = np.unique( np.round( cuv_nodes.std(axis=0) / mesh_coord.std(axis=0), -1 ) )
    if var_ratio > 0:
        # cuv_nodes /= var_ratio
        mesh_coord *= var_ratio
    
    
    matching_index_set=[]
    cuv_values=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        matching_rows = np.all(np.abs(mesh_coord - row) < 1e-3, axis=1)
        if(np.sum(matching_rows) > 0):
            matching_index = int(np.where(matching_rows)[0])
            matching_value = mesh_value[matching_index]
        else:
            matching_index = np.nan
            matching_value = np.nan
        
        matching_index_set.append( matching_index )
        cuv_values.append( matching_value )
        
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    print( len(cuv_nodes) )
    print( len(cuv_values) )

    
    import pandas as pd
        
    nodes_df = pd.DataFrame( np.hstack((cuv_nodes, cuv_values_2d)), columns=["x","y","z",fname] )
    edges_df = pd.DataFrame( cuv_edges, columns=["x","y","z"] )
    
    points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName(fname)
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue(row[fname])
        
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    # png_mesh_GetArray(polydata, "DF")
        
    png_writevtk(polydata, filename=path_out, type="polydata")
    














def png_match_vtk2vtk(path_vtk, path_cuvcut2vtk, path_out, level=1):

    mesh = png_readvtk(path_vtk, type="polydata")
    
    mesh_cuv = png_readstl(path_cuvcut2vtk)
    
    cuv_nodes = png_mesh_GetCoord(mesh_cuv)
    
    mesh_value = png_mesh_GetArray(mesh)
    mesh_coord = png_mesh_GetCoord(mesh)
    
    var_ratio = np.unique( np.round( cuv_nodes.std(axis=0) / mesh_coord.std(axis=0), 0 ) )
    # cuv_nodes /= var_ratio
    mesh_coord *= var_ratio
    
    matching_index_set=[]
    cuv_values=[]
    print(cuv_nodes.shape)
    for row in cuv_nodes:
        matching_rows = np.all(np.abs(mesh_coord - row) < 1e-3, axis=1)
        matching_index = int(np.where(matching_rows)[0])
        matching_value = mesh_value[matching_index]
        
        matching_index_set.append( matching_index )
        cuv_values.append( matching_value )
        
    cuv_values_2d = np.array(cuv_values)[:, np.newaxis]
    
    print( len(cuv_nodes) )
    print( len(cuv_values) )
    
    import pandas as pd
        
    nodes_df = pd.DataFrame( np.hstack((cuv_nodes, cuv_values_2d)), columns=["x","y","z","value"] )
    edges_df = pd.DataFrame( cuv_edges, columns=["x","y","z"] )
    
    points = vtk.vtkPoints()
    values = vtk.vtkFloatArray()
    values.SetName("Values")
    
    for index, row in nodes_df.iterrows():
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        values.InsertNextValue(row['value'])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values)
    
    # 삼각형 셀 생성
    cells = vtk.vtkCellArray()
    
    for index, row in edges_df.iterrows():
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, row['x'])
        triangle.GetPointIds().SetId(1, row['y'])
        triangle.GetPointIds().SetId(2, row['z'])
        cells.InsertNextCell(triangle)
    
    polydata.SetPolys(cells)
    
    
    png_writevtk(polydata, filename=path_out, type="polydata")
    






def png_combine_polydata(polydata_list, col_names):
    """
    polydata_list = [png_readvtk(x) for x in path_out_list]
    col_names = map_list2
    """
    import pandas as pd
    import numpy as np
    
    combined_polydata = polydata_list[0]
    
    if len( np.unique( [ pd.DataFrame(png_mesh_GetCoord(x)).shape[0] for x in polydata_list ] ) ) != 1:
        print( 'Coordinates of polydata sets are different to each other !')
        print( [ pd.DataFrame(png_mesh_GetCoord(x)).shape[0] for x in polydata_list ] )
    
    
    """
    ENUM = enumerate(polydata_list)
    index, polydata = next(ENUM)
    """
    for index, polydata in enumerate(polydata_list):
            
        fname = col_names[index]
        
        values = vtk.vtkFloatArray()
        values.SetName(fname)
        """
        x = png_mesh_GetArray2(polydata, fname)[0]
        png_mesh_GetArray2(polydata_list[0], "VoltMap")
        png_mesh_GetArray2(polydata_list[1], "DF")
        
        import matplotlib.pyplot as plt
        plt.scatter(png_mesh_GetArray2(polydata_list[0], "VoltMap"),
                    png_mesh_GetArray2(polydata_list[1], "DF"), alpha=0.5)
        plt.show()
        np.corrcoef(png_mesh_GetArray2(polydata_list[0], "VoltMap"),
                    png_mesh_GetArray2(polydata_list[1], "DF"))
        
        
        np.array(polydata_list[0].GetPointData().GetArray("x"))[:5]
        np.array(polydata_list[0].GetPointData().GetArray("y"))[:5]
        np.array(polydata_list[0].GetPointData().GetArray("z"))[:5]
        np.array(polydata_list[0].GetPointData().GetArray("VoltMap"))[:5]
        
        """
        for x in png_mesh_GetArray2(polydata, fname):
            values.InsertNextValue(x)
        
        combined_polydata.GetPointData().AddArray(values)
    
    
    return combined_polydata




def png_remove_nontriangle(mesh):
    """Extract triangular faces from vtkPolyData. Return the Nx3 numpy.array of the faces (make sure there are only triangles)."""
    
    m = mesh.GetNumberOfCells()
    
    for i in range(m):
        ptIDs = vtk.vtkIdList()
        mesh.GetCellPoints(i, ptIDs)
        if ptIDs.GetNumberOfIds() != 3:
            mesh.DeleteCell(i)
            
    mesh.RemoveDeletedCells()
    
    return mesh





def convert_cell_data_to_point_data(polydata):
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(polydata)
    cell_to_point.Update()
    return cell_to_point.GetOutput()





