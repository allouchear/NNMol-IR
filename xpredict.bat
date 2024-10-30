set PATH=%PATH%;C:\Users\allouche\Softwares\Anaconda3;C:\Users\allouche\Softwares\Anaconda3\Scripts
set NNMOLDIR=C:\Users\allouche\ML\NNMol-IR\NNMol-IR-main
set PYTHONPATH=%PYTHONPATH%:%NNMOLDIR%
call activate

set fn=%1
set charge=%2
set ms=%3

set Models=%NNMOLDIR%\Models
python %NNMOLDIR%\predict.py --input_file_name=%fn%.xyz --charge=%charge% --multiplicity=%ms% --list_models=%Models%\train_4895_1_0 --average=1 
move ir.csv %fn%.csv

