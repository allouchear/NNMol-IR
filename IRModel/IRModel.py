from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from Utils.UtilsFunctions import *
from IRModel.IRModelNet import *

from tensorflow.keras.layers import Layer

class IRModel(tf.keras.Model):
	def __str__(self):
		return str(self._irModel)

	def __init__(self,
		F=None,                           #dimensionality of feature vector
		K=None,                           #number of radial basis functions
		cutoff=None,                      #cutoff distance for short range interactions
		num_scc=0,                        #number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		  #  Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         #  number of hidden layers in elemental modes block
		num_blocks=5,                     #number of building blocks to be stacked
		num_residual_atomic=2,            #number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       #number of residual layers for refinement of message vector
		num_residual_output=1,            #number of residual layers for the output blocks
		num_frequencies=2000,             #number of frequencies
                drop_rate=None,                   #initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   #activation function
		output_activation_fn=tf.nn.relu,  # output activation function
		dtype=tf.float32,                 #single or double precision
		loss_type='SID',                  # loss type (SID, MAE)
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",             # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		nn_model="MPNN",		  # MPNN (Message-Passing Neural network), EANN (Embedded Atom Neural Network), EAMP (Embedded Atom Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		seed=None):
		super().__init__(dtype=dtype, name="IRModel")

		self._irModel=None

		self._num_outputs=num_frequencies+1
		self._irModel=IRModelNet (  nn_model=nn_model,
						F=F,
						K=K,
						num_scc =  num_scc,
						em_type =  em_type,
						num_hidden_nodes_em =  num_hidden_nodes_em,
						num_hidden_layers_em = num_hidden_layers_em,
						cutoff=cutoff,
						dtype=dtype, 
						num_blocks=num_blocks, 
						num_residual_atomic=num_residual_atomic,
						num_residual_interaction=num_residual_interaction,
						num_residual_output=num_residual_output,
						num_outputs=self.num_outputs, # Qa + frequencies
						activation_fn=activation_fn,
						output_activation_fn=output_activation_fn,
						drop_rate=drop_rate,
						nhlambda=nhlambda,
						basis_type=basis_type,
						loss_type=loss_type,
						seed=seed)

	def computeProperties(self, data):
		return self._irModel.computeProperties(data)

	def computeLoss(self, data):
		return self._irModel.computeLoss(data)

	def print_parameters(self):
		self._irModel.print_parameters()

	def __call__(self, data, closs=True):
		return self._irModel(data,closs=closs)

	@property
	def irModel(self):
		return self._irModel

	@property
	def dtype(self):
		return self._irModel.dtype

	@property
	def neuralNetwork(self):
		return self._irModel.neuralNetwork

	@property
	def nhlambda(self):
		return self._irModel.nhlambda

	@property
	def cutoff(self):
		return self._irModel.cutoff

	@property
	def num_outputs(self):
		return self._num_outputs
