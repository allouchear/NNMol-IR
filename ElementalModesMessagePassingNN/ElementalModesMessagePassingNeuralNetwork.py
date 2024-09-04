import tensorflow as tf
from tensorflow.keras.layers import Layer
from .RBFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from .ElementalModesBlock import *
from Utils.ActivationFunctions import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

#returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
def scaled_charges(Z, Qa, Q_tot=None, mol_idx=None): # Qa not Qal including for GFN1
	if mol_idx is None:
		mol_idx = tf.zeros_like(Z)
	#number of atoms per batch (needed for charge scaling)
	Na_per_batch = tf.math.segment_sum(tf.ones_like(mol_idx, dtype=Qa.dtype), mol_idx)
	if Q_tot is None: #assume desired total charge zero if not given
		Q_tot = tf.zeros_like(Na_per_batch, dtype=Qa.dtype)
	#return scaled charges (such that they have the desired total charge)
	return Qa + tf.gather(((Q_tot-tf.math.segment_sum(Qa, mol_idx))/Na_per_batch), mol_idx)

#returns scaled atomic spin charges to reproduce the molecule spin charges
def NSE(Z, QaA, QaB, faA, faB, QAlpha_mol=None, QBeta_mol=None, mol_idx=None): # Q_mol=None => mol charge = 0
	if mol_idx is None:
		mol_idx = tf.zeros_like(Z) # all atoms in one molecule
	#number of atoms per batch (needed for charge scaling)
	Na_per_batch = tf.math.segment_sum(tf.ones_like(mol_idx, dtype=QaA.dtype), mol_idx)
	if QAlpha_mol is None: #assume desired total charge zero if not given
		QAlpha_mol = tf.zeros_like(Na_per_batch, dtype=QaA.dtype)
	if QBeta_mol is None: #assume desired total charge zero if not given
		QBeta_mol = tf.zeros_like(Na_per_batch, dtype=QaB.dtype)
	#return scaled charges (such that they have the desired total charge)
	QaAlpha = QaA + faA*tf.gather(((QAlpha_mol-tf.math.segment_sum(QaA, mol_idx))/tf.math.segment_sum(faA, mol_idx)), mol_idx)
	QaBeta  = QaB + faB*tf.gather(((QBeta_mol -tf.math.segment_sum(QaB, mol_idx))/tf.math.segment_sum(faB, mol_idx)), mol_idx)
	return QaAlpha, QaBeta



class ElementalModesMessagePassingNeuralNetwork(Layer):
	def __str__(self):
		st =    "Elemental Modes Message Passing Neural Network\n"+\
			"----------------------------------------------"\
			+"\n"+\
			"Dimensionality of feature vector                       : " + str(self.F)\
			+"\n"+str(self.rbf_layer)\
			+"\n"+\
			"Elemental modes type                                   : " + str(self.em_type) + " ; 0 => only Z"+ \
			"\n                                                             1 => Z + Masses" + \
			"\n                                                             2 => Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)" \
			+"\n"+\
			"Number of hidden layers in element modes block         : " + str(len(self.elemental_modes_block.hidden_layers)) \
			+"\n"+\
			"Number of hidden nodes by layer in element modes block : " + str(self.elemental_modes_block.hidden_layers[0].get_config()['units']) \
			+"\n"+\
			"Number of building blocks                              : " + str(self.num_blocks)\
			+"\n"+\
			"Number of residuals for interaction                    : " + str(len(self.interaction_block[0].interaction.residual_layer))\
			+"\n"+\
			"Number of residuals for atomic                         : " + str(len(self.interaction_block[0].residual_layer))\
			+"\n"+\
			"Number of outputs                                      : " + str(self.num_outputs)\
			+"\n"+\
			"Number of residuals for outputs                        : " + str(len(self.output_block[0].residual_layer))\
			+"\n"+\
			"Float type                                             : " + str(self.dtype)\

		if self.activation_fn is None:
			st = "\nActivation function                                     : None"
		else:
			st += "\nActivation function                                    : "+str(self.activation_fn.__name__)
		if self.output_activation_fn is None:
			st = "\nOutput activation function                              : None"
		else:
			st += "\nOutput activation function                             : "+str(self.output_activation_fn.__name__)

		return st

	def __init__(self,
		F,                               #dimensionality of feature vector
		K,                               #number of radial basis functions
		cutoff,                          #cutoff distance for short range interactions
		num_scc=0,                       #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		 #elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None, 	 #number of hidden nodes by layer in element modes block , None => F
		num_hidden_layers_em = 2, 	 #number of hidden layer in element modes block
		num_blocks=5,                    #number of building blocks to be stacked
		num_residual_atomic=2,           #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,      #number of residual layers for refinement of message vector
		num_residual_output=1,           #number of residual layers for the output blocks
		num_outputs=2001,      		 #number of outputs by atom
                drop_rate=None,                  #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,  #activation function
		output_activation_fn=tf.nn.relu, # output activation function
		basis_type="Default",            #radial basis type : GaussianNet (Default), Gaussian, Bessel, Slater, 
		beta=0.2,			 #for Gaussian basis type
		dtype=tf.float32,                #single or double precision
		seed=None):
		super().__init__(dtype=dtype, name="ElementalModesMessagePassingNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype
		self._num_scc=num_scc

		self._num_outputs = num_outputs 

		self._F = F
		self._K = K
		self._em_type = em_type
		self._cutoff = tf.constant(cutoff,dtype=dtype) #cutoff for neural network interactions
		
		self._activation_fn = activation_fn
		self._output_activation_fn = output_activation_fn

		#drop rate regularization
		"""
		if drop_rate is None:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=False)
		else:
			self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=True)
		"""
		self._drop_rate = tf.Variable(0.0, shape=[], name="drop_rate",dtype=dtype,trainable=False)


		#elemental_modes_block blocks and output layers
		self._elemental_modes_block = ElementalModesBlock(F, num_hidden_nodes=num_hidden_nodes_em, num_hidden_layers=num_hidden_layers_em, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype, name="elemental_modes_block")

		#radial basis function expansion layer
		self._rbf_layer = RBFLayer(K,  self._cutoff, beta=beta, basis_type=basis_type, name="rbf_layer",dtype=dtype)

		#embedding blocks and output layers
		self._interaction_block = []
		self._output_block = []
		nouts = num_outputs 
		if num_scc > 0:
			nouts += 4 # for QaAlpha, faAlpha, QaBeta & faBeta : the 4 first outputs

		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, name="InteractionBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(F, nouts, num_residual_output, activation_fn=activation_fn, name="OutputBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))

	def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
		#calculate interatomic distances
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
			Rj += offsets
		Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
		return Dij

	def get_input_elements(self, Z, M, QaAlpha, QaBeta):

		#QaAlpha = None
		#QaBeta = None

		Z =tf.Variable(Z,dtype=self.dtype)
		Z = tf.reshape(Z,[Z.shape[0],1])
		if M is not None and self.em_type >=1 :
			M =tf.Variable(M,dtype=self.dtype)
			M = tf.reshape(M,[M.shape[0],1])
		if QaAlpha is not None and self.em_type >=2 :
			QaAlpha =tf.Variable(QaAlpha,dtype=self.dtype)
			QaAlpha = tf.reshape(QaAlpha,[QaAlpha.shape[0],1])
		if QaBeta is not None and self.em_type >=2 :
			QaBeta =tf.Variable(QaBeta,dtype=self.dtype)
			QaBeta = tf.reshape(QaBeta,[QaBeta.shape[0],1])
		#print("QaAlpha=",QaAlpha)
		#print("QaBeta=",QaBeta)
		#print("M=",M)

		f = None
		if M is not None and QaAlpha is not None and  QaBeta is not None and self.em_type >=2:
			f=tf.concat([Z,M,QaAlpha, QaBeta],1)
		elif M is not None and self.em_type >=1:
			f=tf.concat([Z,M],1)
		else:
			f=tf.concat([Z],1)

		return f

	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, Z, R, idx_i, idx_j, M=None, QaAlpha=None, QaBeta=None, offsets=None, mol_idx=None):
		#calculate distances (for long range interaction)
		Dij = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)

		#calculate radial basis function expansion
		rbf = self.rbf_layer(Dij)
		#print("rbf=\n",rbf,"\n-------------------------------------\n")

		#initialize feature vectors according to embeddings for nuclear charges
		#print("Z=",Z)
		#print("f=",f)

		QaA = tf.Variable(QaAlpha, dtype=self.dtype)
		QaB = tf.Variable(QaBeta, dtype=self.dtype)
		faA = tf.ones([len(QaAlpha)],dtype=self.dtype)
		faB = tf.ones([len(QaBeta)],dtype=self.dtype)
		QAlpha_mol = tf.math.segment_sum(QaA, mol_idx)
		QBeta_mol  = tf.math.segment_sum(QaB, mol_idx)
		#print("QAlpha_mol=",QAlpha_mol)
		#print("QBeta_mol=",QBeta_mol)
		ibegin=0
		if self.num_scc>0:
			ibegin=4
		for j in range(self.num_scc+1):
			f = self.get_input_elements(Z, M, QaA, QaB)
			x = self.elemental_modes_block(f)
			#print("x=",x)
			outputs = 0
			#print("outputs=",outputs)
			nhloss = 0 #non-hierarchicality loss
			for i in range(self.num_blocks):
				x = self.interaction_block[i](x, rbf, idx_i, idx_j)
				out = self.output_block[i](x)
				outputs += out
				#compute non-hierarchicality loss
				if j==self.num_scc:
					out2 = out[ibegin:]**2
					if i > 0:
						nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
					lastout2 = out2

			if self.num_scc>0:
				QaA = outputs[:,0]
				faA = outputs[:,1]
				QaB = outputs[:,2]
				faB = outputs[:,3]
				QaA, QaB = NSE(Z, QaA, QaB, faA, faB, QAlpha_mol=QAlpha_mol, QBeta_mol=QBeta_mol, mol_idx=mol_idx)
				#print("j=",j,"QAlpha_mol=", tf.math.segment_sum(QaA, mol_idx))

		if self.num_scc>0:
			outputs = outputs[:,ibegin:]

		#print("outputsAll=",outputs)
		#apply scaling/shifting
		#print("outputs=",outputs)

		#outputs = tf.constant(newout)
		#print("outputsShape=",outputs.shape)
		#print("outputsType=",type(outputs))
		out0=outputs[:,0:1]
		# Intensities must be >=0
		#outo=tf.keras.activations.relu(outputs[:,1:], alpha=0.0, max_value=None, threshold=0.0)
		outo=self.output_activation_fn(outputs[:,1:])
		#print("outOShape=",outo.shape)
		#print("out0Shape=",out0.shape)
		outputs=tf.concat([out0, outo], 1)
		#print("outShape=",outputs.shape)
		#print("outputs=",outputs)
		return outputs, Dij, nhloss

	@property
	def drop_rate(self):
		return self._drop_rate

	@property
	def num_scc(self):
		return self._num_scc
    
	@property
	def num_blocks(self):
		return self._num_blocks

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def dtype(self):
		return self._dtype

	@property
	def elemental_modes_block(self):
		return self._elemental_modes_block

	@property
	def F(self):
		return self._F

	@property
	def K(self):
		return self._K

	@property
	def em_type(self):
		return self._em_type

	@property
	def cutoff(self):
		return self._cutoff

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def output_activation_fn(self):
		return self._output_activation_fn
    
	@property
	def rbf_layer(self):
		return self._rbf_layer

	@property
	def interaction_block(self):
		return self._interaction_block

	@property
	def output_block(self):
		return self._output_block

