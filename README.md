#NNMol-IR : A program for computing Infrared spectrum form 3D molecular structure.
=========================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Requirement
 - tensorflow, 
 - tensorflow_probability, 
 - tensorflow_addons, 
 - nvidia-tensorrt

After installation of conda, and activation of your environnement,  Type : 
```console
pip install tensorflow==2.12.0
pip install tensorflow_probability==0.19.0
pip install tensorflow_addons==0.20.0
pip3 install nvidia-tensorrt
```

## Installation

Using git, under a terminal, type : 
```console
git clone https://github.com/allouchear/NNMol-IR.git
```
You can also download the .zip file of NNMP-IR : Click on Code and Download ZIP

## Main programs

### train.py
**Using the database in Data (h5 format) directory, make the training**\
see train.inp and ./xtrain bash script.
Please note that in this example, the IR data used are from a scaled DFT, not from experiment. In the published paper, we used the experimental data. However, we cannot share these values because they are from a commercial database. You can purchase them from NIST. See our published paper for more details.

### evaluation.py
**Test the models (one or an ensemble of models) using a database**\
see  ./xevaluation bash script in example directory.

### predict.py
**Predict the IR spectrum using a xyz file**\
see ./xpredict\* bash scripts.


## Contributors
The code is written by Abdul-Rahman Allouche.

## License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).

