#from __future__ import absolute_import
#import os
import tensorflow as tf
import sys
import os
import argparse
import logging
import string
import random
from datetime import datetime
import numpy  as np
from Utils.PeriodicTable import *
from Utils.PhysicalConstants import *


def convert_indexed_slices_to_tensor(idx_slices):
    return tf.scatter_nd(tf.expand_dims(idx_slices.indices, 1),
                         idx_slices.values, idx_slices.dense_shape)

def print_gradients_norms(gradients,trainable_weights,details=False):
	norm=0
	maxgr=[None,None]
	nVars=0
	for g,tr in zip(gradients,trainable_weights):
		if isinstance(g, tf.IndexedSlices) is True:
			g=convert_indexed_slices_to_tensor(g)

		#print("tr=",tr)
		#print("g=",g)
		if g is None:
			#print("g=",g)
			#print("tr=",tr)
			g = tr*0.0 # to be remove

		g2=g*g
		ng2=tf.reduce_sum(g2)
		nV = tf.math.count_nonzero(tf.math.greater_equal(abs(g), 1e-14))
		nVars = nVars + nV
		if details==True:
			print("name=",tr.name, "shape=", g2.shape," norm=",tf.sqrt(ng2).numpy())
		norm += ng2
		if maxgr[0] is None or ng2>maxgr[1]:
			maxgr =[tr.name, ng2]
		#if tr.name=="drop_rate:0":
		#	print("name=",tr.name, "tr=", tr.numpy(), "g=",g.numpy())
		#if tr.name=="InteractionBlock_0/Hidden_0/kernel:0":
		#	print("name=",tr.name, "tr=", tr.numpy(), "g=",g.numpy())
		#if tr.name=="InteractionBlock_0/Hidden_0/bias:0":
		#	print("name=",tr.name, "tr=", tr.numpy(), "g=",g.numpy())
		#if tr.name=="bf_layerexpansion coefficients:0":
		#	print("name=",tr.name, "tr16=", tr.numpy()[1][6], "tr11=",tr.numpy()[1][1], "tr61=",tr.numpy()[6][1], "tr66=",tr.numpy()[6][6])


	if maxgr[0] is not None:
		print("name=",maxgr[0], " normMax=",tf.sqrt(maxgr[1]).numpy())
	print("norm=",tf.sqrt(norm).numpy())
	print("#Variables=", nVars.numpy())

def getArguments():
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
	parser.add_argument("--nn_model", default=None, type=str,   help="nn_model: MPNN (Message-Passing Neural network), EMMPNN (Element Modes Message Passing NN)")
	parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
	parser.add_argument("--num_scc", type=int,    default=0, help="number of cycles of updating of spin atomic charges (0 => no cycle)")
	parser.add_argument("--em_type", type=int,    default=0, help="Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)")
	parser.add_argument("--num_hidden_nodes_em", type=int, default=None,  help="number of nodes on each hidden layer in elemental modes block. Default = num_features")
	parser.add_argument("--num_hidden_layers_em", type=int,   default=2, help="number of hidden layers in elemental modes block; Default=2")
	parser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
	parser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
	parser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
	parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
	parser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
	parser.add_argument("--num_frequencies", type=int,  default=2000, help="number of frequencies")
	parser.add_argument("--dataFrameNameFrequencies", type=str,  default="frequenciesConv", help="name of frequencies in the data frame (default frequenciesConv)")
	parser.add_argument("--dataFrameNameIntensities", type=str,  default="irIntensitiesConv", help="name of frequencies in the data frame (default irIntensitiesConv)")
	parser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
	parser.add_argument("--dataset", type=str,   help="file path to dataset")
	parser.add_argument("--num_train", type=int,   help="number of training samples")
	parser.add_argument("--num_valid", type=int,   help="number of validation samples")
	parser.add_argument("--num_test", type=int, default=-1,  help="number of test samples. Default : nAll-num_valid-num_train")
	parser.add_argument("--seed", default=42, type=int,   help="seed for splitting dataset into training/validation/test")
	parser.add_argument("--max_steps", type=int,   help="maximum number of training steps")
	parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate used by the optimizer")
	parser.add_argument("--max_norm", default=1000.0, type=float, help="max norm for gradient clipping")
	parser.add_argument("--ema_decay", default=0.999, type=float, help="exponential moving average decay used by the trainer")
	parser.add_argument("--drop_rate", default=None, type=float, help="drop_rate probability for dropout regularization of rbf layer")
	parser.add_argument("--l2lambda", type=float, help="lambda multiplier for l2 loss (regularization)")
	parser.add_argument("--nhlambda", type=float, help="lambda multiplier for non-hierarchicality loss (regularization)")
	parser.add_argument("--decay_steps", type=int, help="decay the learning rate every N steps by decay_rate")
	parser.add_argument("--decay_rate", type=float, help="factor with which the learning rate gets multiplied by every decay_steps steps")
	parser.add_argument("--batch_size", type=int, help="batch size used per training step")
	parser.add_argument("--valid_batch_size", type=int, help="batch size used for going through validation_set")
	parser.add_argument('--charge_weight', default=1.0, type=float, help="this defines the charge contribution to the loss function")
	parser.add_argument('--atomic_charge_weight', default=0.0, type=float, help="this defines the atomic charge contribution to the loss function")
	parser.add_argument('--dipole_weight', default=1.0, type=float, help="this defines the dipole contribution to the loss function")
	parser.add_argument('--summary_interval', type=int, help="write a summary every N steps")
	parser.add_argument('--validation_interval', type=int, help="check performance on validation set every N steps")
	parser.add_argument('--save_interval', type=int, help="save progress every N steps")
	parser.add_argument('--conv_distance_to_bohr', default=1.0, type=float, help="convert coefficient of distance in data to Bohr")
	parser.add_argument("--dtype", default='float64', type=str,   help="type of float : float32 or float64")
	parser.add_argument("--loss_type", default='SID', type=str,   help="type of loss : SID, MAE (default SID)")
	parser.add_argument("--learning_schedule_type", default='EXP', type=str,   help="type of  type of learning rate schedule : Exp, None, Time, Plateau (default Exp)")
	parser.add_argument("--patience", default=5, type=int,   help="Patience for Plateau schedule (default 5)")
	parser.add_argument("--min_lr", default=0, type=float,   help="min learning rate for Plateau schedule (default 0)")
	parser.add_argument("--restart_optimizer_nsteps", default=0, type=int,   help="restart optimizer after n steps (Default 0 => No restart)")

	parser.add_argument("--use_average", default=0, type=int,   help="use averaged values of weigths (default 0)")
	parser.add_argument("--verbose", default=0, type=int,   help="print verbose : 0 = Min")
	parser.add_argument("--basis_type", default="Default", type=str,   help="radial basis type : GaussianNet (Default for MPNN), Gaussian, Bessel, Slater")
	parser.add_argument("--activation_function", default='shifted_softplus', type=str,   help="Name of activation function : None, shifted_softplus, softplus, scaled_shifted_softplus, self_normalizing_shifted_softplus, smooth_ELU, self_normalizing_smooth_ELU, self_normalizing_asinh, self_normalizing_tanh, tanh, elu, ... or any tensorflow ones, default=shifted_softplus")
	parser.add_argument("--output_activation_function", default='relu', type=str,   help="Name of activation function : None, shifted_softplus, softplus, scaled_shifted_softplus, self_normalizing_shifted_softplus, smooth_ELU, self_normalizing_smooth_ELU, self_normalizing_asinh, self_normalizing_tanh, tanh, elu, ... or any tensorflow ones, default=relu")

	#if no command line arguments are present, config file is parsed
	config_file='config.txt'
	fromFile=False
	if len(sys.argv) == 1:
		fromFile=True
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

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def setOutputLocationFiles(args):
	#create directories
	#a unique directory name is created for this run based on the input
	if args.restart is None:
		directory=datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + id_generator() +"_F"+str(args.num_features)+"K"+str(args.num_basis)+"b"+str(args.num_blocks)+"a"+str(args.num_residual_atomic)+"i"+str(args.num_residual_interaction)+"o"+str(args.num_residual_output)+"nf"+str(args.num_frequencies)+"cut"+str(args.cutoff)+"l2"+str(args.l2lambda)+"nh"+str(args.nhlambda)+"drop"+str(args.drop_rate)
	else:
		directory=args.restart

	logfile= os.path.join(directory, 'train.log')
	print("creating directories...")
	
	if not os.path.exists(directory):
		os.makedirs(directory)
	logging.basicConfig(filename=logfile,level=logging.DEBUG)
	logging.info("creating directories...")
	best_dir = os.path.join(directory, 'best')
	if not os.path.exists(best_dir):
		os.makedirs(best_dir)
	average_dir = os.path.join(directory, 'average')
	if not os.path.exists(best_dir):
		os.makedirs(average_dir)
	log_dir = os.path.join(directory, 'logs')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	metrics_dir = os.path.join(directory, 'metrics')
	if not os.path.exists(metrics_dir):
		os.makedirs(metrics_dir)
	best_checkpoint = os.path.join(best_dir, 'best.ckpt')
	average_checkpoint = os.path.join(average_dir, 'average.ckpt')
	step_checkpoint = os.path.join(log_dir,  'model.ckpt')
	best_loss_file  = os.path.join(best_dir, 'best_loss.npz')

	print("Main output directory          : ", directory)
	print("Loggin file                    : ", logfile)
	print("Loggin file directory          : ", log_dir)
	print("Best variables checkpoint      : ", best_checkpoint)
	print("Average variables checkpoint   : ", average_checkpoint)
	print("Step checkpoint                : ", step_checkpoint)
	print("Best loss file                 : ", best_loss_file)
	
	return directory, log_dir, best_dir, average_dir,  metrics_dir, best_checkpoint, average_checkpoint, step_checkpoint, best_loss_file

def _factorial(n):
        return tf.exp(tf.math.lgamma(float(n+1)))

factorialData = None
nfactorialData=0
def factorial(n):
	global factorialData
	global nfactorialData 
	if factorialData is None:
		nfactorialData=10
		factorialData = [1]*(nfactorialData+1)
		for i in range(2,nfactorialData+1):
			factorialData[i] = factorialData[i-1]*i

	if n<=nfactorialData:
		return factorialData[n]
	else:
        	return _factorial(n)

def _doubleFactorial(n):
	#return tf.exp(tf.math.lgamma(float(n)/2+1.))*tf.math.sqrt(tf.math.pow(2.0,n+1.0)/pi)
	v=1
	if n%2==0:
		for i in range(2,n+1,2):
			v *= i 
	else:
		for i in range(3,n+1,2):
			v *= i 
	return v

dfactorialData = None
ndfactorialData=0
def doubleFactorial(n):
	#return _doubleFactorial(n)
	if n<=1:
		return 1
	global dfactorialData
	global ndfactorialData 
	if dfactorialData is None:
		ndfactorialData=10
		dfactorialData = [1]*(ndfactorialData+1)
		dfactorialData[0] = 1
		for i in range(2,ndfactorialData+1,2):
			dfactorialData[i] = dfactorialData[i-2]*i

		dfactorialData[1] = 1
		for i in range(3,ndfactorialData+1,2):
			dfactorialData[i] = dfactorialData[i-2]*i

	if n<=ndfactorialData:
		#print(" dd = ", n, dfactorialData[n], _doubleFactorial(n))
		return dfactorialData[n]
	else:
		return _doubleFactorial(n)


def _binomial(i,j, dtype=tf.float32):
	return factorial(i)/factorial(j)/factorial(i-j)

binomialData = None
nbinomialData=0
def binomial(i,j, dtype=tf.float32):
	#return _binomial(i,j)
	if i<=1:
		return 1
	global binomialData
	global nbinomialData 
	if binomialData is None:
		nbinomialData=10
		binomialData = np.empty([nbinomialData+1, nbinomialData+1])
		binomialData[0][0] = 1
		for k in range(nbinomialData+1):
			for l in range(k+1):
				binomialData[k][l] = _binomial(k,l,dtype=dtype)

	if i<=nbinomialData:
		return binomialData[i][j]
	else:
		return _binomial(i,j,dtype=dtype)

def save_xyz(data, fileName):
		#print(data)
		Q_tot=data['Q']
		Z=data['Z']
		R=tf.Variable(data['R'])
		mol_idx = data['mol_idx']
		if mol_idx is None:
			mol_idx = tf.zeros_like(Z)
		index=tf.constant(range(len(mol_idx)),dtype=tf.int32)
		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(mol_idx), mol_idx)
		if Q_tot is None: #assume desired total charge zero if not given
			Q_tot = tf.zeros_like(Na_per_batch)

		f = open(fileName,"w")
		periodicTable=PeriodicTable()
		nb = 0
		for imol in range(len(Q_tot)):
			nAtoms=int(Na_per_batch[imol])
			f.write(str(nAtoms)+"\n")
			f.write("Coordinates in Angstrom\n")
			ne=nb+nAtoms
			Rmol = tf.gather(R, index[nb:ne])
			Zmol = tf.gather(Z, index[nb:ne])
			Rmol *= 0.52917721
			for ia in range(len(Rmol)):
				symbol=periodicTable.elementZ(int(Zmol[ia].numpy())).symbol
				f.write('{:<6s} '.format(symbol))
				for c in range(3):
					f.write('{:>20.14f} '.format(Rmol[ia][c].numpy()))
				f.write("\n")
			nb=ne
		f.close()

def save_spectra(data, fileName):
		#print(data)
		Z=data['Z']
		I=tf.Variable(data['I'])
		F=tf.Variable(data['F'])
		mol_idx = data['mol_idx']
		if mol_idx is None:
			mol_idx = tf.zeros_like(Z)
		print("mol_idx=",mol_idx)
		index=tf.constant(range(len(mol_idx)),dtype=tf.int32)
		#number of atoms per batch
		Na_per_batch = tf.math.segment_sum(tf.ones_like(mol_idx), mol_idx)

		f = open(fileName,"w")
		nb = 0
		for imol in range(len(Na_per_batch)):
			nAtoms=int(Na_per_batch[imol])
			ne=nb+nAtoms
			Fmol = tf.gather(F, index[nb:ne])
			Imol = tf.gather(I, index[nb:ne])
			for ia in range(len(Fmol)):
				f.write('{:>20.14f}\t{:>20.14f}\n'.format(Fmol[ia].numpy(),Imol[ia].numpy()))
				f.write("\n")
			nb=ne
		f.close()

def remove_directory(directory):
	for root, dirs, files in os.walk(directory):
		for file in files:
			os.remove(os.path.join(root, file))
	os.removedirs(directory)

# Coefficient of determination, called also R_qsuared
"""
The coefficient of determination measures how well the predicted values match (and not just follow) the observed values. 
It depends on the distance between the points and the 1:1 line (and not the best-fit line). 
Closer the data to the 1:1 line, higher the coefficient of determination
"""
def coefficient_of_determination(y, y_pred):
	residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
	total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
	if tf.math.abs(residual)<1e-14:
		R2 = tf.Variable(1.0,dtype=y.dtype)
	else:
		R2 = 1.0-residual/(total+1e-14)

	return R2

def pearson_correlation_coefficient(y, ypred):
	y=tf.convert_to_tensor(y)
	ypred=tf.cast(ypred,y.dtype)
	ymean = tf.reduce_mean(y)
	ypredmean = tf.reduce_mean(ypred)
	yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean))
	ypredvar = tf.reduce_sum( tf.math.squared_difference(ypred, ypredmean))
	cov = tf.reduce_sum( (y - ymean) * (ypred - ypredmean))
	den = tf.sqrt(yvar*ypredvar)
	if den<1e-14:
		corr = tf.Variable(1.0,dtype=y.dtype)
	else:
		corr = cov/tf.sqrt(yvar*ypredvar+1e-14)

	#return tf.constant(1.0,dtype=y.dtype)-corr
	return corr

def pearson_correlation_coefficient_squared(y, ypred):
	return tf.math.square(pearson_correlation_coefficient(y,ypred))

def save_model_parameters(args, minF=400, maxF=4000.0):
	try:
		directory=args.restart
		fileName=os.path.join(directory, 'model_parameters.txt')
		f= open(fileName,"w")
		#define command line arguments
		#f.write("--directory="+args.restart+"\n")
		f.write("--nn_model="+str(args.nn_model)+"\n")
		f.write("--num_features="+str(args.num_features)+"\n")
		f.write("--num_scc="+str(args.num_scc)+"\n")
		f.write("--em_type="+str(args.em_type)+"\n")
		f.write("--num_hidden_nodes_em="+str(args.num_hidden_nodes_em)+"\n")
		f.write("--num_hidden_layers_em="+str(args.num_hidden_layers_em)+"\n")
		f.write("--num_basis=" +str(args.num_basis)+"\n")
		f.write("--num_blocks=" +str(args.num_blocks)+"\n")
		f.write("--num_residual_atomic=" +str(args.num_residual_atomic)+"\n")
		f.write("--num_residual_interaction="+str(args.num_residual_interaction)+"\n")
		f.write("--num_residual_output=" +str(args.num_residual_output)+"\n")
		f.write("--num_frequencies=" +str(args.num_frequencies)+"\n")
		f.write("--min_frequencies=" +str(minF)+"\n")
		f.write("--max_frequencies=" +str(maxF)+"\n")
		f.write("--cutoff=" +str(args.cutoff)+"\n")
		f.write("--dtype="+args.dtype+"\n")
		f.write("--loss_type="+args.loss_type+"\n")
		f.write("--use_average="+str(args.use_average)+"\n")
		f.write("--basis_type="+str(args.basis_type)+"\n")
		f.write("--activation_function="+str(args.activation_function)+"\n")
		f.write("--output_activation_function="+str(args.output_activation_function)+"\n")
		f.close()
	except Exception as Ex:
		print("Write Failed.", Ex)
		raise Ex

def read_model_parameters(directory):
	fileName= os.path.join(directory, 'model_parameters.txt')
	#define command line arguments
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument("--directory", type=str, default=None,  help="training directory")
	parser.add_argument("--nn_model", default=None, type=str,   help="nn_model: MPNN (Message-Passing Neural network) , EMMPNN (Element Modes Message Passing NN)")
	parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
	parser.add_argument("--num_scc", type=int,    default=0, help="number of cycles of updating of spin atomic charges (0 => no cycle)")
	parser.add_argument("--em_type", type=int,    default=0, help="Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)")
	parser.add_argument("--num_hidden_nodes_em", type=int, default=None,  help="number of nodes on each hidden layer in elemental modes block. Default = num_features")
	parser.add_argument("--num_hidden_layers_em", type=int,   default=2, help="number of hidden layers in elemental modes block; Default=2")
	parser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
	parser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
	parser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
	parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
	parser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
	parser.add_argument("--num_frequencies", type=int,  default=2000, help="number of frequencies")
	parser.add_argument("--min_frequencies", type=float,  default=400, help="min of frequencies")
	parser.add_argument("--max_frequencies", type=float,  default=4000, help="max of frequencies")
	parser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
	parser.add_argument("--dtype", default='float64', type=str,   help="type of float : float32 or float64")
	parser.add_argument("--basis_type", default="Default", type=str,   help="radial basis type : GaussianNet (Default for MPNN), Gaussian, Bessel, Slater")
	parser.add_argument("--verbose", default=0, type=int,   help="print verbose : 0 = Min")
	parser.add_argument("--loss_type", default='SID', type=str,   help="type of loss : SID, MAE (dfault SID)")
	parser.add_argument("--learning_schedule_type", default='EXP', type=str,   help="type of  type of learning rate schedule : Exp, None, Time (default Exp)")
	parser.add_argument("--patience", default=5, type=int,   help="Patience for Plateau schedule (default 5)")
	parser.add_argument("--min_lr", default=0, type=float,   help="min learning rate for Plateau schedule (default 0)")
	parser.add_argument("--restart_optimizer_nsteps", default=0, type=int,   help="restart optimizer after n steps (Default 0 => No restart)")
	parser.add_argument("--use_average", default=0, type=int,   help="use averaged values of weigths (default 0)")
	parser.add_argument("--activation_function", default='shifted_softplus', type=str,   help="Name of activation function : None, shifted_softplus, softplus, scaled_shifted_softplus, self_normalizing_shifted_softplus, smooth_ELU, self_normalizing_smooth_ELU, self_normalizing_asinh, self_normalizing_tanh, tanh, elu, ... or any tensorflow ones")
	parser.add_argument("--output_activation_function", default='relu', type=str,   help="Name of activation function : None, shifted_softplus, softplus, scaled_shifted_softplus, self_normalizing_shifted_softplus, smooth_ELU, self_normalizing_smooth_ELU, self_normalizing_asinh, self_normalizing_tanh, tanh, elu, ... or any tensorflow ones, default=relu")

	try:
		args = parser.parse_args(["@"+fileName])
	except Exception as Ex:
		print("I cannot read parameters.", Ex)
		raise Ex

	setattr(args, 'directory', directory)
		
	return args

