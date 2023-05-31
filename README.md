# png.LA_flattening
Author: Kipoong Kim (kkp7700@gmail.com)

## About
This project builds on the existing "[LA_flattening](https://github.com/martanunez/LA_flattening)" method described in:
[*Fast quasi-conformal regional flattening of the left atrium*. Marta Nuñez-Garcia, Gabriel Bernardino, Francisco Alarcón, Gala Caixal, Lluís Mont, Oscar Camara, and Constantine Butakoff.  IEEE Transactions on Visualization and Computer Graphics (2020)](https://ieeexplore.ieee.org/abstract/document/8959311). Please cite this reference when using this code. 
The "LA_flattening" method uses a Windows program called "FillSurfaceHoles" to conduct the surface reconstruction for the clipped PV region (hole). 
However, since "FillSurfaceHoles" is no longer compatible, this project uses pymeshlab's "screened poisson surface reconstruction" method.

## Code
[Python](https://www.python.org/) scripts depending (basically) on [VTK](https://vtk.org/) and [VMTK](http://www.vmtk.org/). 


## How to use

### Materials
Folder list
- cuv
- DF
- LAT
- PS
- Smax
- VoltMap
- thickness
- Miscellaneous mapping data

### Process

(1) Convert .cuv file (generated from CUVIA software[^1]) to .vtk file
(2) Slice the PVs in .vtk files using the [paraview](https://www.paraview.org/)
(3) Apply surface reconstruction on the sliced .vtk files.
(4) Perform the LA_flattening method

[^1] Kim, I. S., Lim, B., Shim, J., Hwang, M., Yu, H. T., Kim, T. H., ... & CUVIA-AF1 Investigators. (2019). Clinical usefulness of computational modeling-guided persistent atrial fibrillation ablation: updated outcome of multicenter randomized study. Frontiers in Physiology, 10, 1512.



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
conda create -n vmtk_env38 python=3.8 itk vtk vmtk scipy xlsxwriter matplotlib
conda activate vmtk_env38
```

Then, install other packages:
```
conda install pandas numpy matplotlib pymeshlab
```





## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
