# NNMol-IR : A Machine learning program for computing Infrared spectrum form 3D molecular structure.
=====================================================================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Requirement
 - tensorflow, 
 - tensorflow_probability, 
 - tensorflow_addons, 
 - nvidia-tensorrt (if you want to use GPU)

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
The data is not included in the GitHub repository. You have to download it from Zenodo: [doi.org/10.5281/zenodo.13681778](https://doi.org/10.5281/zenodo.13681778)

### evaluation.py
**Test the models (one or an ensemble of models) using a database**\
see  ./xevaluation bash script in example directory.
Please note that the models are not included in the GitHub repository. You have to download them from Zenodo: [doi.org/10.5281/zenodo.13681778](https://doi.org/10.5281/zenodo.13681778)

### predict.py
**Predict the IR spectrum using a xyz file**\
see ./xpredict\* bash scripts.
Please note that the models are not included in the GitHub repository. You have to download them from Zenodo: [doi.org/10.5281/zenodo.13681778](https://doi.org/10.5281/zenodo.13681778)

## Citations

Please cite : Neural Network Approach for Predicting Infrared Spectra from 3D Molecular Structure, Chemical Physics Letters xxx, xxx (2024). [https://doi.org/10.1016/j.cplett.2024.141603](https://doi.org/10.1016/j.cplett.2024.141603)

title = {Neural network approach for predicting infrared spectra from 3D molecular structure},
journal = {Chemical Physics Letters},
pages = {141603},
year = {2024},
issn = {0009-2614},
doi = {https://doi.org/10.1016/j.cplett.2024.141603},
url = {https://www.sciencedirect.com/science/article/pii/S0009261424005451},
author = {Saleh {Abdul Al} and Abdul-Rahman Allouche},
keywords = {Infrared Spectra, Machine Learning, Neural Network, DFT scaled frequencies},
abstract = {We developed a machine learning (ML) model to directly predict IR spectra from three-dimensional (3D) molecular structures. The spectra predicted by our model significantly outperform those from density functional theory (DFT) calculations, even after scaling. In a test set of 200 molecules, our model achieves a Spectral Information Similarity Metric (SIS) of 0.92 surpassing the value achieved by DFT scaled frequencies which is 0.57. Additionally, our model considers anharmonic effects offering a fast alternative to laborious anharmonic calculations. Moreover, our model can be used to predict various types of spectra (as UV or NMR) as a function of molecular structure.}
}

## Contributors
The code is written by Abdul-Rahman Allouche.

## License
This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).

