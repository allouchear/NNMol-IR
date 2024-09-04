import tensorflow as tf
from tensorflow.keras.layers import Layer
from .RBFLayer import *
from .InteractionBlock import *
from .OutputBlock      import *
from Utils.ActivationFunctions import *

def softplus_inverse(x):
	'''numerically stable inverse of softplus transform'''
	return x + np.log(-np.expm1(-x))

class MessagePassingNeuralNetwork(Layer):
	def __str__(self):
		st =    "Message Passing Neural Network\n"+\
			"------------------------------"\
			+"\n"+\
			"Dimensionality of feature vector    : " + str(self.F)\
			+"\n"+str(self.rbf_layer)\
			+"\n"+\
			"Number of building blocks           : " + str(self.num_blocks)\
			+"\n"+\
			"Number of residuals for interaction : " + str(len(self.interaction_block[0].interaction.residual_layer))\
			+"\n"+\
			"Number of residuals for atomic      : " + str(len(self.interaction_block[0].residual_layer))\
			+"\n"+\
			"Number of outputs                   : " + str(self.num_outputs)\
			+"\n"+\
			"Number of residuals for outputs     : " + str(len(self.output_block[0].residual_layer))\
			+"\n"+\
			"Float type                          : " + str(self.dtype)\

		if self.activation_fn is None:
			st = "\nActivation function                  : None"
		else:
			st += "\nActivation function                 : "+str(self.activation_fn.__name__)
		if self.output_activation_fn is None:
			st = "\nOutput activation function           : None"
		else:
			st += "\nOutput activation function          : "+str(self.output_activation_fn.__name__)

		return st

	def __init__(self,
		F,                               #dimensionality of feature vector
		K,                               #number of radial basis functions
		cutoff,                          #cutoff distance for short range interactions
		num_blocks=5,                    #number of building blocks to be stacked
		num_residual_atomic=2,           #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,      #number of residual layers for refinement of message vector
		num_residual_output=1,           #number of residual layers for the output blocks
		num_outputs=2001,      		 #number of outputs by atom
                drop_rate=None,                  #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,  #activation function
		output_activation_fn=tf.nn.relu, # output activation function
		basis_type="Default",            # radial basis type : GaussianNet (Default), Gaussian, Bessel, Slater, 
		beta=0.2,			 # for Gaussian basis type
		dtype=tf.float32,                #single or double precision
		seed=None):
		super().__init__(dtype=dtype, name="MessagePassingNeuralNetwork")

		assert(num_blocks > 0)
		assert(num_outputs > 0)
		self._num_blocks = num_blocks
		self._dtype = dtype
		self._num_outputs = num_outputs
		self._F = F
		self._K = K
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

		#atom embeddings (we go up to Pu(94), 95 because indices start with 0)
		self._embeddings = tf.Variable(tf.random.uniform([95, self.F], minval=-tf.math.sqrt(tf.cast(3.0,dtype=dtype)), maxval=tf.math.sqrt(tf.cast(3.0,dtype=dtype))
				   , seed=seed, dtype=dtype), name="embeddings", dtype=dtype,trainable=True)
		tf.summary.histogram("embeddings", self.embeddings)  

		#radial basis function expansion layer
		self._rbf_layer = RBFLayer(K,  self._cutoff, beta=beta, basis_type=basis_type, name="rbf_layer",dtype=dtype)

		#embedding blocks and output layers
		self._interaction_block = []
		self._output_block = []
		for i in range(num_blocks):
			self.interaction_block.append(
			InteractionBlock(F, num_residual_atomic, num_residual_interaction, activation_fn=activation_fn, name="InteractionBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))
			self.output_block.append(
				OutputBlock(F, num_outputs, num_residual_output, activation_fn=activation_fn, name="OutputBlock"+str(i),
					seed=seed, drop_rate=self.drop_rate, dtype=dtype))

	def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
		#calculate interatomic distances
		Ri = tf.gather(R, idx_i)
		Rj = tf.gather(R, idx_j)
		if offsets is not None:
			Rj += offsets
		Dij = tf.sqrt(tf.nn.relu(tf.reduce_sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
		return Dij

	#calculates the atomic properties and distances (needed if unscaled charges are wanted e.g. for loss function)
	def atomic_properties(self, Z, R, idx_i, idx_j, M=None, QaAlpha=None, QaBeta=None, offsets=None, mol_idx=None):
		#calculate distances (for long range interaction)
		Dij = self.calculate_interatomic_distances(R, idx_i, idx_j, offsets=offsets)

		#print("Z=\n",Z,"\n-------------------------------------\n")
		#print("R=\n",R,"\n-------------------------------------\n")
		#print("idx_i=\n",idx_i,"\n-------------------------------------\n")
		#print("Dij=\n",Dij,"\n-------------------------------------\n")
		#calculate radial basis function expansion
		rbf = self.rbf_layer(Dij)
		#print("rbf=\n",rbf,"\n-------------------------------------\n")

		#initialize feature vectors according to embeddings for nuclear charges
		x = tf.gather(self.embeddings, Z)

		#print("x=\n",x,"\n-------------------------------------\n")

		#apply blocks
		#outputs = tf.zeros([x.shape[0],self._num_outputs], dtype=x.dtype)
		outputs = 0
		#print("outputs=",outputs)
		nhloss = 0 #non-hierarchicality loss
		for i in range(self.num_blocks):
			x = self.interaction_block[i](x, rbf, idx_i, idx_j)
			out = self.output_block[i](x)
			outputs += out
			#compute non-hierarchicality loss
			out2 = out**2
			if i > 0:
				nhloss += tf.reduce_mean(out2/(out2 + lastout2 + 1e-7))
			lastout2 = out2
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
		#print("out act func=",self.output_activation_fn)
		#print("min outo=",tf.reduce_min(outo))
		#print("max outo=",tf.reduce_max(outo))

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
	def num_blocks(self):
		return self._num_blocks

	@property
	def num_outputs(self):
		return self._num_outputs

	@property
	def dtype(self):
		return self._dtype

	@property
	def embeddings(self):
		return self._embeddings

	@property
	def F(self):
		return self._F

	@property
	def K(self):
		return self._K

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

