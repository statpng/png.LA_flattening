# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:34:27 2023
@author: statpng

https://github.com/statpng/png.LA_flattening
"""


import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


os.getcwd()
os.chdir("D:/data/1.Mesh3D/png.LA_flattening")
    
from png_functions import *
from aux_functions import *
from clip_aux_functions import *





# path 내에 저장된 *.plt 파일명에 *-Voltage.plt로 변경하는 코드
path = "D:/data/map_test/Voltage"
for file in os.listdir(path):
    """
    file = os.listdir(path)[0]
    """
    if file.endswith(".plt"):
        id_part, ext_part = file.split(".", 1)
        new_name_part = "-Voltage.plt"
        new_filename = f"{id_part}{new_name_part}"
        os.rename(os.path.join(path, file), os.path.join(path, new_filename))

        print(f"Renaming {file} to {new_filename}")





# 1. CUV to VTK

R_gsub(r"D:\data\map22")
path_data = "D:/data/map22"
path_cuv = path_data + "/cuv"


ID = ID_list[0]  # '2846-outputXML_.cuv'
ID_list = [ x for x in os.listdir(path_cuv) if len(re.findall(r"\.cuv", x)) > 0 ]
for ID in ID_list:
    print(ID)
    
    filename = re.findall(r'\d+', ID)[0]
    filename_cuv = filename + "-cut"
        
    png_cuvia2polydata(path=path_cuv + "/" + ID, \
                       outpath=path_cuv + "/" + filename_cuv + ".vtk", \
                       level=1)







# 2. Slice the PVs in VTK

path_data = "D:/data/map22"
path_cuv = path_data + "/cuv"

ID_list = [ x for x in os.listdir(path_cuv) if len(re.findall(r"cut2.vtk", x)) > 0 ]

for ID in ID_list:
    """
    ID = ID_list[0]
    """
    print(ID)
    
    if not os.path.isfile(path_cuv + "/" + ID[:-4] + "_poly.vtk"):
        png_unstructured2polydata(path_cuv + "/" + ID)





# 3. Apply LA flattening

path_data = "D:/data/map22"
path_cuv = path_data + "/cuv"

ID_list = [ x for x in os.listdir(path_cuv) if len(re.findall(r"poly.vtk", x)) > 0 ]

# If you have a list of IDs to be excluded
ExcludeList = [str(x) for x in [2851, 2856, 2857, 2877, 2884]]
ID_list = sorted( list(set( [item for item in ID_list if not any(re.search(z, item) for z in ExcludeList)] )) )



""" print(ID_list)
 '2826-cut2_poly.vtk'
 '2846-cut2_poly.vtk'
 '2854-cut2_poly.vtk'
 '2878-cut2_poly.vtk'
 '2919-cut2_poly.vtk'
 '2922-cut2_poly.vtk'
 '2995-cut2_poly.vtk'
 '3001-cut2_poly.vtk'
 '3072-cut2_poly.vtk'
 '3076-cut2_poly.vtk'
 '3081-cut2_poly.vtk'
 '3093-cut2_poly.vtk'
 '3094-cut2_poly.vtk'
 '3136-cut2_poly.vtk'
 '3184-cut2_poly.vtk'
 '3247-cut2_poly.vtk'
 '3297-cut2_poly.vtk'
 '3362-cut2_poly.vtk'
 '3438-cut2_poly.vtk'
 '3442-cut2_poly.vtk'
"""

"""
ID = ID_list[0]
"""


filename = re.findall(r'\d+', ID)[0]
subname = ID.split("-")[1][:-4]

filename_flat = filename+"-"+subname
path_flat = path_data+"/flat"

savepath_flat = path_flat+"/"+filename_flat
if not os.path.exists(savepath_flat):
    os.mkdir(savepath_flat)


meshfile = path_cuv+"/"+filename_flat+".vtk"
meshfile_open = savepath_flat+"/"+filename_flat+"_crinkle_clipped.vtk" 
meshfile_open_no_mitral = savepath_flat+"/"+filename_flat+"_clipped_mitral.vtk" 
meshfile_closed = savepath_flat+"/"+filename_flat+"_clipped_c.vtk"









# Auto-region data integration

import os, re
from png_functions import *
import pandas as pd


path_auto = "D:/data/map22/region_vm"
file_list = [ x for x in os.listdir(path_auto) if len(re.findall(r"\.", x)) == 0 ]
ID_list = [x[:4] for x in file_list]

idx1 = 0
for idx1 in range(len(file_list)):
    ID = ID_list[idx1]
    
    dir_path = path_auto+"/" + file_list[idx1]
    filename_plt_list = [ dir_path + "/" + x for x in os.listdir( dir_path ) if len(re.findall(r"R\d+.plt", x))>0 ]
    
    node_df_total = pd.DataFrame([])
    for idx2 in range(len(filename_plt_list)):
        filename_plt = filename_plt_list[idx2]
        region_name = re.findall(r"(R\d+).plt", filename_plt)[0]
        
        node_df = png_plt2node(filename_plt)
        node_df["region"] = region_name 
        
        node_df_total = pd.concat([node_df_total, node_df], axis=0)
    
    node_df_total["ID"] = ID
    
    node_df_total.to_csv(path_auto+"/df_region_"+ID+".csv", index=False)










# 4. Integration of mapping data

path_data = "D:/data/map22"
path_flat = path_data+"/flat"

map_list = ["VoltMap","DF","LAT","Smax", "PS", "thickness"][:-1]

ID_list = [ x for x in os.listdir(path_flat) if os.path.exists(path_data+"/flat/"+x+"/"+x+"_clipped_c_flat.vtk") ]

for ID in ID_list:
    """
    ID = ID_list[0]
    """
    
    print(ID)
    
    if not os.path.isfile(path_data+"/flat"+"/"+ID+"/"+ID+"_clipped_c_flat.vtk"):
        continue
    
    filename = re.findall(r'\d+', ID)[0]
    subname = ID.split("-")[1]
    fullname = filename + "-" + subname
    savepath_flat = path_flat+"/"+fullname
    
    final_flat = savepath_flat+"/"+fullname+"_clipped_c_flat.vtk"
    path_final_flat = os.path.dirname(final_flat)
    
    path_out_list = [ final_flat[:-4]+"-"+x+'.vtk' for x in map_list ]
    path_combined = final_flat[:-4]+"-combined.vtk"
    
    if os.path.isfile(path_combined):
        continue
        
    x = map_list[0]
    for x in map_list:
        type_vtk = "polydata"
        if x == "DF":
            type_vtk = "unstructured"
        
        
        MappingDataPath = path_data+'/'+x+'/'+[y for y in os.listdir(path_data+'/'+x) if filename in y][0]
        path_plt = MappingDataPath[:-4]+".plt"
        path_vtk = MappingDataPath[:-4]+".vtk"
        
        if not os.path.isfile(path_vtk):
            png_plt2polydata(path_plt, path_vtk, fname=x)
    
        path_out = final_flat[:-4]+"-"+x+".vtk"
        
        png_match_flat2vtk(path_vtk, final_flat, path_out, type_vtk)
    
    
    
    map_list2 = ["VoltMap", "DF", "LAT", "SmaxMap", "PSoutput", "thickness"][:-1]
    combined_polydata = png_combine_polydata( [png_readvtk(x) for x in path_out_list], map_list2)
    # png_mesh_GetArray(combined_polydata)
    writevtk(combined_polydata, filename=path_combined)
    
    
    
    combined_polydata_region = png_match_region( combined_polydata, path_auto+"/df_region_"+filename+".csv" )
    
    png_mesh_GetArray(combined_polydata_region)
    
    writevtk(combined_polydata_region, filename=path_combined[:-4]+"-region.vtk")
    
    png_writePointArray(combined_polydata_region, filename=final_flat[:-4]+"_values.csv")
    print(ID)







# (node, edge) 데이터를 vtk로 변환하는 코드
path_reconst = "D:/data/map22/reconst"
filename = "df_flat-3442-cut2_poly-reconst_selprob"
filenames = ["Table-reconst-VoltMap-"+x for x in ["2826", "2846"]]

for filename in filenames:
    path_nodes = path_reconst + "/" + filename + "-nodes.csv"
    path_edges = path_reconst + "/" + filename + "-edges.csv"
    path_vtk = path_reconst + "/" + filename + ".vtk"
    
    png_read_and_write_coord(path_nodes, path_edges, path_vtk)







