from __future__ import absolute_import
import os
import tensorflow as tf
from MessagePassingNN.MessagePassingNeuralNetwork import *
from Utils.ActivationFunctions import *
from Utils.UtilsFunctions import *
from Utils.UtilsLoss import *
from .NeuralNetwork import *

from tensorflow.keras.layers import Layer

class IRModelNet(tf.keras.Model):
	def __str__(self):
		st = str(self.neuralNetwork)
		return st

	def __init__(self,
		F=128,                            #dimensionality of feature vector
		K=64,                             #number of radial basis functions
		cutoff=8.0,                       #cutoff distance for short range interactions (atomic unit)
		num_scc=0,                        #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		  #  Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		num_outputs=2,        		  #number of outputs by atom
                drop_rate=None,                   #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   #activation function
		output_activation_fn=tf.nn.relu,  # output activation function
		dtype=tf.float32,                 #single or double precision
		loss_type='SID',                  # loss type (SID, MAE)
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",             # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		nn_model="MPNN",		  # MPNN (Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		seed=None):
		super().__init__(dtype=dtype, name="IRModelNet")
		self._loss_type = loss_type

		self._neuralNetwork = neuralNetwork(
			nn_model=nn_model,
			F=F,
			K=K,
			num_scc =  num_scc,
			em_type = em_type,
			num_hidden_nodes_em = num_hidden_nodes_em,
			num_hidden_layers_em = num_hidden_layers_em,
			basis_type=basis_type,
			cutoff=cutoff, 
			num_blocks=num_blocks, 
			num_residual_atomic=num_residual_atomic, 
			num_residual_interaction=num_residual_interaction,
			num_residual_output=num_residual_output, 
			num_outputs=num_outputs,
			drop_rate=drop_rate,
			activation_fn=activation_fn,
			output_activation_fn=output_activation_fn,
			dtype=dtype,
			seed=seed) 


		self._nhlambda=nhlambda

		self._dtype=dtype


	def computeProperties(self, data):
		#print(data)
		Z=data['Z']
		M=data['M']
		QaAlpha=data['QaAlpha']
		QaBeta=data['QaBeta']


		R=tf.Variable(data['R'],dtype=self.dtype)
		idx_i=data['idx_i']
		idx_j=data['idx_j']
		#print("-------outputs------------------\n",outputs,"\n----------------------------\n")


		with tf.GradientTape() as g:
			g.watch(R)
			outputs, Dij , nhloss = self.neuralNetwork.atomic_properties(Z, R, idx_i, idx_j, M=M, QaAlpha=QaAlpha, QaBeta=QaBeta, offsets=None, mol_idx=data['mol_idx'])

			#print("outputs shape=",outputs.shape)
			Qa = outputs[:,0]
			I  = outputs[:,1:]
			#print("I shape=",I.shape)
			I = tf.squeeze(tf.math.segment_sum(I, data['mol_idx']))
			#print("I shape=",I.shape)
			#print("data['M'] shape=",tf.constant(data['M'],dtype=self.dtype).shape)
			#print("data['Z'] shape=",tf.constant(data['Z'],dtype=self.dtype).shape)
			#print("data['I'] shape=",tf.constant(data['I'],dtype=self.dtype).shape)
			charges = tf.squeeze(tf.math.segment_sum(Qa, data['mol_idx']))

		return charges, Qa, I, nhloss

	def computeLoss(self, data):
		with tf.GradientTape() as tape:
			charges, Qa, I, nhloss = self.computeProperties(data)
			loss = spectral_loss(I, tf.constant(data['I'],dtype=self.dtype),type=self.loss_type)
			#print("lossAll=",loss)
			#print("LossType=",self.loss_type)
			#print("Ishape=",I.shape)
			#print("nshape=",tf.reshape(I,[-1]).shape[0])
			loss /= tf.reshape(I,[-1]).shape[0]
			#print("lossDiv=",loss)
			if self.nhlambda>0:
				loss += self.nhlambda*nhloss

		gradients = tape.gradient(loss, self.trainable_weights)
		#print("Loss=",loss)
		#print("mean=",mean)
		#print("-------Loss-----------------\n",loss,"\n----------------------------\n")
		#print("-------Gradients------------\n",gradients,"\n----------------------------\n")
		return charges, Qa, I, loss, gradients

	def __call__(self, data, closs=True):
		if closs is not True:
			charges, Qa, I, nhloss = self.computeProperties(data)
			loss=None
			gradients=None
		else:
			charges, Qa, I, loss, gradients = self.computeLoss(data)
			
		return charges, Qa, I, loss, gradients

	def print_parameters(self):
		pass

	@property
	def dtype(self):
		return self._dtype

	@property
	def neuralNetwork(self):
		return self._neuralNetwork

	@property
	def nhlambda(self):
		return self._nhlambda

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def cutoff(self):
		return self.neuralNetwork.cutoff


