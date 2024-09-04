from ase import io
from ase import units
import pandas as pd

import tensorflow as tf
import sys
#tf.config.threading.set_intra_op_parallelism_threads(40)
#tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Predictor import *
import os

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--input_file_name', default=None, type=str, help="Input file name xyz, POSCAR, ....")
	parser.add_argument('--average', default=0, type=int, help="1 to Use average parameters instead best ones, 0 for best parameters")
	parser.add_argument('--charge', type=float, default=0.0, help="Charge of molecule, Default=0.0")
	parser.add_argument('--multiplicity', type=float, default=0.0, help="Charge of molecule, Default=0.0")
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=False
	if len(sys.argv) == 2 and sys.argv[1].find('--') == -1:
		config_file=sys.argv[1]
		fromFile=True

	if fromFile is True:
		print("Try to read configuration from ",config_file, "file")
		if os.path.isfile(config_file):
			args = parser.parse_args(["@"+config_file])
		else:
			args = parser.parse_args(["--help"])
	else:
		args = parser.parse_args()

	return args


args = getArguments()

inputfname=args.input_file_name
lmodels=args.list_models
lmodels=lmodels[0].split(',')
charge=args.charge
multiplicity=args.multiplicity
print(lmodels)

atoms = io.read(inputfname)

print("--------- Input geometry --------------------")
print("Z : " , atoms.get_atomic_numbers())
print("Positions : ",atoms.get_positions())
print("---------------------------------------------")

predictor = Predictor(
		lmodels,
		atoms,
		conv_distance=1/units.Bohr,
		conv_mass=1,
		charge=charge,  
		multiplicity=multiplicity,
		average=args.average>0
		)

F,I= predictor.computeIR()

print(F,I)

df = pd.DataFrame({'#F':F,'I':I})
print(df)
df.to_csv("ir.csv",  index=False, sep=' ')
