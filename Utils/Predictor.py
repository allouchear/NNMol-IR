import tensorflow as tf
import numpy as np
import ase
from IRModel.IRModel import *
from Utils.UtilsFunctions import *
from Utils.PhysicalConstants import *

'''
Calculator for the atomic simulation environment (ASE)
It computes IR intensities using a NNMol
'''

def getExampleData():
	data = {}
	data['Z'] =  [1,6]
	data['M'] =  [1,6]
	data['Mult'] =  1
	data['Q'] =  0
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['mol_idx'] =   tf.zeros_like(data['Z'])
	data['offsets'] =   None
	data['QAlpha'] = 0
	data['QBeta'] = 0
	data['QaAlpha'] = [0,0]
	data['QaBeta'] = [0,0]
	return data

def get_data_from_atoms(atoms, charge=0, multiplicity=1, conv_distance=1.0/BOHR_TO_ANG, conv_mass=1.0):
		nAtoms = len(atoms)
		nnAtoms = nAtoms*(nAtoms-1)
		idx_i = np.zeros([nnAtoms], dtype=int)
		idx_j = np.zeros([nnAtoms], dtype=int)
		offsets = np.zeros([nnAtoms,3], dtype=float)
		count = 0
		for i in range(nAtoms):
			for j in range(nAtoms):
				if i != j:
					idx_i[count] = i
					idx_j[count] = j
					count += 1
		# build data from Z,R,idx_i,idx_j,mol_idx
		data = {}
		data['Z']     =  atoms.get_atomic_numbers()
		data['R']     =  atoms.get_positions()*conv_distance
		data['Q']     =  charge
		data['Mult']  =  multiplicity 
		data['M']     =  atoms.get_masses()*conv_mass # in amu to set masses near Z
		data['idx_i'] =  idx_i
		data['idx_j'] =  idx_j
		data['mol_idx'] =   tf.zeros_like(data['Z'])
		data['offsets'] =   offsets
		data['QAlpha'] = 0.5*(charge-multiplicity+1)
		data['QBeta'] = 0.5*(charge-multiplicity-1)
		N=len(data['Z'])
		data['QaAlpha']= [data['QAlpha']/N]*N
		data['QaBeta']= [data['QBeta']/N]*N

		return data
	
class Predictor:
	def __init__(self,
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		atoms,                            #ASE atoms object
		charge=0,                         #system charge
		multiplicity=1,                   #system multiplicity
		conv_distance=ANG_TO_BOHR,        #coef. conversion of distance from unit of ase in unit of NNMol
		conv_mass=1.0,                    #coef. conversion of distance to amu
		average=False, 
		):

		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories

		self._conv_distance = conv_distance
		self._conv_mass     = conv_mass
		self._charge        = charge
		self._multiplicity  = multiplicity
		self._average = average
		
		self._models = []

		n=0
		num_frequencies=None
		min_frequencies=None
		max_frequencies=None
		for directory in self._models_directories:
			args = read_model_parameters(directory)
			print("directory=",directory)
			if self._average:
				average_dir = os.path.join(directory, 'average')
				checkpoint = os.path.join(average_dir, 'average.ckpt')
			else:
				best_dir = os.path.join(directory, 'best')
				checkpoint = os.path.join(best_dir, 'best.ckpt')

			irModel=   IRModel (F=args.num_features,
			K=args.num_basis,
			cutoff=args.cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_scc=args.num_scc,
			em_type=args.em_type,
			num_hidden_nodes_em=args.num_hidden_nodes_em,
			num_hidden_layers_em=args.num_hidden_layers_em,
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			num_frequencies=args.num_frequencies,
			activation_fn=activation_deserialize(args.activation_function),
			output_activation_fn=activation_deserialize(args.output_activation_function),
			nn_model=args.nn_model,
			basis_type=args.basis_type,
			loss_type=args.loss_type,
			)
			if num_frequencies is None:
				num_frequencies=args.num_frequencies
				max_frequencies=args.max_frequencies
				min_frequencies=args.min_frequencies
			elif num_frequencies != args.num_frequencies:
				print("**********************************************************")
				print("Error number of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			elif abs(max_frequencies-args.max_frequencies)>1e-2:
				print("**********************************************************")
				print("Error max of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			elif abs(min_frequencies-args.min_frequencies)>1e-2:
				print("**********************************************************")
				print("Error min of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			

			data = getExampleData()
			charges, Qa, I, loss, gradients = irModel(data,closs=False) # to set auto shape
			#print("checkpoint=",checkpoint)
			irModel.load_weights(checkpoint)
			self._models.append(irModel)
		self._num_frequencies  = num_frequencies
		self._min_frequencies  = min_frequencies
		self._max_frequencies  = max_frequencies
		self._atoms  = atoms

	def _computeProperties(self):
		data = None
		n=0
		nModel=len(self._models)
		atoms= self._atoms
		for i in range(nModel):
			if i==0 or self._models[i].cutoff != self._models[i-1].cutoff:
				data = get_data_from_atoms(atoms, conv_distance=self._conv_distance, conv_mass=self._conv_mass)
			molcharges, atomcharges, I, nhloss = self._models[i].computeProperties(data)
			#print("atomcharges=",atomcharges)

			if i == 0:
				self._molcharges = molcharges
				self._atomcharges  = atomcharges
				self._I  = I
			else:
				n = i+1

				if atomcharges is not None:
					self._atomcharges += (atomcharges-self._atomcharges)/n
				if I is not None:
					self._I +=  (I-self._I)/n 
				if molcharges is not None:
					self._molcharges += (molcharges-self._molcharges)/n


		self._I = self._I.numpy()

	def computeIR(self):
		self._computeProperties()
		return self.get_fequencies(),  self._I

	def get_intensities(self):
		self._computeProperties(atoms)
		return self._I

	def get_fequencies(self):
		return np.linspace(self.min_frequencies, self.max_frequencies, self.num_frequencies,endpoint=False)

	@property
	def I(self):
		return self._I

	@property
	def cutoff(self):
		return self._cutoff

	@property
	def model(self):
        	return self._models

	@property
	def checkpoint(self):
		return self._checkpoint

	@property
	def Z(self):
		return self._Z

	@property
	def charge(self):
		return self._charge

	@property
	def multiplicity(self):
		return self._multiplicity

	@property
	def R(self):
		return self._R

	@property
	def offsets(self):
		return self._offsets

	@property
	def idx_i(self):
		return self._idx_i

	@property
	def idx_j(self):
		return self._idx_j

	@property
	def conv_distance(self):
		return self._conv_distance

	@property
	def conv_mass(self):
		return self._conv_mass

	@property
	def num_frequencies(self):
		return self._num_frequencies

	@property
	def min_frequencies(self):
		return self._min_frequencies

	@property
	def max_frequencies(self):
		return self._max_frequencies
