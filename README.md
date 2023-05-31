# png.LA_flattening
Author: Kipoong Kim (kkp7700@gmail.com)

## About
This project builds on the existing "[LA_flattening](https://github.com/martanunez/LA_flattening)" method described in:
[*Fast quasi-conformal regional flattening of the left atrium*. Marta Nuñez-Garcia, Gabriel Bernardino, Francisco Alarcón, Gala Caixal, Lluís Mont, Oscar Camara, and Constantine Butakoff.  IEEE Transactions on Visualization and Computer Graphics (2020)](https://ieeexplore.ieee.org/abstract/document/8959311). Please cite this reference when using this code. 
The "LA_flattening" method uses a Windows program called "FillSurfaceHoles" to conduct the surface reconstruction for the clipped PV region (hole). 
However, since "FillSurfaceHoles" is no longer compatible, this project uses pymeshlab's "screened poisson surface reconstruction" method instead.

[Python](https://www.python.org/) scripts depending (basically) on [VTK](https://vtk.org/) and [VMTK](http://www.vmtk.org/). 


## How to use

### Materials
- cuv (.cuv or .vtk)
- DF (.vtk)
- LAT (.vtk)
- PS (.plt or .vtk)
- Smax (.vtk)
- VoltMap (.vtk)
- thickness (.vtk)
- Miscellaneous mapping data (.vtk)

### Process

(1) Convert .cuv file (generated from CUVIA software[^1]) to .vtk file:  `ID.cuv >> ID-cut.vtk`

(2) Slice the PVs in .vtk files using the [paraview](https://www.paraview.org/):  `ID-cut.vtk >> ID-cut2.vtk`

(3) Convert unstructured polydata to structured polydata:  `ID-cut2.vtk >> ID-cut2_poly.vtk`

(4) Apply surface reconstruction on the sliced .vtk files:  `ID-cut2_poly.vtk >> ID-cut2_poly-fill.stl >> ID-cut2_poly-fill.vtk`

(5) Perform the LA_flattening method:  `./flat/ID-cut2_poly/ID-cut2_poly_clipped_c_flat.vtk`

(6) Matching the mapping data to the flat:  `./flat/ID-cut2_poly_clipped_c_flat-${map}.vtk`  for  map=Voltage, DF, Smax, and etc.

(7) Integrate the mapping data sets:  `./flat/ID-cut2_poly_clipped_c_flat-combined.vtk`

(7) Incorporating the auto-region data into the dataset:  `./flat/ID-cut2_poly_clipped_c_flat-combined-region.vtk`


[^1]: Kim, I. S., Lim, B., Shim, J., Hwang, M., Yu, H. T., Kim, T. H., ... & CUVIA-AF1 Investigators. (2019). Clinical usefulness of computational modeling-guided persistent atrial fibrillation ablation: updated outcome of multicenter randomized study. Frontiers in Physiology, 10, 1512.



## Dependencies
The scripts in this repository were successfully run with:
1. Windows 10
    - [Python](https://www.python.org/) 3.8.16
    - [VMTK](http://www.vmtk.org/) 1.5.0
    - [VTK](https://vtk.org/) 9.1.0
  
Other required packages are: NumPy, SciPy, xlsxwriter, Matplotlib, joblib, python-tk, Pandas, and pymeshlab.




### Python packages installation
To install VMTK follow the instructions [here](http://www.vmtk.org/download/). The easiest way is installing the VMTK [conda](https://docs.conda.io/en/latest/) package (it additionally includes VTK, NumPy, etc.). It is recommended to create an environment where VMTK is going to be installed and activate it:

```
conda create -n env_vmtk38 python=3.8 itk vtk vmtk scipy xlsxwriter matplotlib
conda activate env_vmtk38
```

Then, install other packages:
```
pip install pandas numpy matplotlib pymeshlab pyvista
```





## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
