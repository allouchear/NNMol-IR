#!/home/theochem/allouche/Softwares/anaconda3/bin/python -u
import tensorflow as tf
#tf.config.threading.set_intra_op_parallelism_threads(40)
#tf.config.threading.set_inter_op_parallelism_threads(40)

import numpy as np
from Utils.Evaluator import *
#from Utils.UtilsTrain import *
import os
import argparse


def buildH5File(data_file,cas_file, metrics_dir):
	if not os.path.exists(metrics_dir):
		os.makedirs(metrics_dir)
	dcas = pd.read_csv(cas_file,header=None)
	#print("CAS =", dcas[0])
	outFile = os.path.join(metrics_dir,  'DataCAS.h5')

	casToGet = dcas[0].to_numpy()
	df = pd.read_hdf(data_file)
	print("Shape of original data file = ", df.shape)
	'''
	print(df.index)
	print("df cas to Get=")
	for cas in casToGet:
		print(df.loc[cas])
	'''

	df = df.loc[casToGet]
	print("Shape of new data file = ", df.shape)
	'''
	print("Check if cas are taken: ")
	n=0
	for cas in casToGet:
		if cas in df.index:
			print(cas)
			n += 1
	print(df.shape)
	'''
	df.to_hdf(outFile, 'df')
	print("New data file created , see ", outFile)

	return outFile

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('--data_file', default="Data/data.h5", type=str, help="Input file name h5")
	parser.add_argument('--average', default=0, type=int, help="1 to Use average parameters instead best ones, 0 for best parameters")
	parser.add_argument('--list_models', type=str, nargs='+', help="list of directory containing fitted models (at least one file), ....")
	parser.add_argument('--metrics_dir', default="metrics_evaluation", type=str, help="directory where result are saved")
	parser.add_argument('--cas_file', default="none", type=str, help="the name of file containing the cas of molecules, default = None")
	parser.add_argument('--batch_size', default=32,type=int, help="batch size, default=32")
	parser.add_argument("--dataFrameNameFrequencies", type=str,  default="frequenciesConv", help="name of frequencies in the data frame (default frequenciesConv)")
	parser.add_argument("--dataFrameNameIntensities", type=str,  default="irIntensitiesConv", help="name of frequencies in the data frame (default irIntensitiesConv)")

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

if "NONE" in args.cas_file.upper():
	dataFile = args.data_file
else:
	dataFile = buildH5File(args.data_file, args.cas_file, args.metrics_dir)

lmodels=args.list_models
lmodels=lmodels[0].split(',')
metrics_dir=args.metrics_dir
batch_size=args.batch_size
dataFrameNameFrequencies=args.dataFrameNameFrequencies
dataFrameNameIntensities=args.dataFrameNameIntensities
print("Models = ", lmodels)
print("DataFile = ", dataFile)
print("Metrcis directory = ", metrics_dir)
print("Batch size = ", batch_size)

evaluator = Evaluator(
		lmodels,
		dataFile=dataFile,
		nvalues=-1,  # -1 for all values in datfile
		#nvalues=2, # 2 for test 
		batch_size=batch_size,
		convDistanceToBohr=1.0, # conv data to unit of NNMol
		dataFrameNameFrequencies=dataFrameNameFrequencies,
		dataFrameNameIntensities=dataFrameNameIntensities,
		average=args.average>0
		)

print("Accuraties for :", evaluator.nvalues, "values")
print("---------------------------------------------")
acc=evaluator.computeAccuracies(verbose=True)
#print_all_acc(acc)
print("Save metrics in :", metrics_dir)
fileNames=evaluator.saveAnalysis(metrics_dir)

