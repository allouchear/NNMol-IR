from __future__ import absolute_import
import os
import tensorflow as tf
from MessagePassingNN.MessagePassingNeuralNetwork import *
from ElementalModesMessagePassingNN.ElementalModesMessagePassingNeuralNetwork import *
from Utils.ActivationFunctions import *

def neuralNetwork(
		F=128,                            # dimensionality of feature vector
		K=64,                             # number of radial basis functions
		cutoff=8.0,                       # cutoff distance for short range interactions (atomic unit)
		num_scc=0,                        # number of cycles of updating of spin atomic charges (0 => no cycle)
		em_type = 0,       		  # Elemental modes type : 0(default) => only Z, 1=> Z + Masses, 2=> Z+Masses+QaAlpha+QaBeta (so we use charge &multiplicity)
		num_hidden_nodes_em = None,       # number of nodes on each hidden layer in elemental modes block
		num_hidden_layers_em = 2,         # number of hidden layers in elemental modes block
		num_blocks=5,                     # number of building blocks to be stacked
		num_residual_atomic=2,            # number of residual layers for atomic refinements of feature vector
		num_residual_interaction=3,       # number of residual layers for refinement of message vector
		num_residual_output=1,            # number of residual layers for the output blocks
		num_outputs=2,        		  # number of outputs by atom
                drop_rate=None,                   # initial value for drop rate (None=No drop)
		activation_fn=shifted_softplus,   # activation function
		output_activation_fn=tf.nn.relu,  # output activation function
		dtype=tf.float32,                 # single or double precision
		nhlambda=0,			  # lambda multiplier for non-hierarchicality loss (regularization)
		basis_type="Default",             # radial basis type : GaussianNet (Default for MPNN), Gaussian(Default for EANN), Bessel, Slater, 
		nn_model="MPNN",		  # MPNN (Message-Passing Neural network), EANN (Embedded Atom Neural Network), EAMP (Embedded Atom Message-Passing Neural network), EANNP (Embedded Atom Pairs Neural Network)
		seed=None):

	neuralNetwork = None

	if nn_model=="EMMPNN":
		neuralNetwork = ElementalModesMessagePassingNeuralNetwork(
		F,
		K,cutoff, 
		num_scc =  num_scc,
		em_type = em_type,
		num_hidden_nodes_em=num_hidden_nodes_em,
		num_hidden_layers_em=num_hidden_layers_em,
		num_blocks=num_blocks, num_residual_atomic=num_residual_atomic, 
		num_residual_interaction=num_residual_interaction,
		num_residual_output=num_residual_output, num_outputs=num_outputs,
		drop_rate=drop_rate,
		basis_type=basis_type,
		activation_fn=activation_fn,
		output_activation_fn=output_activation_fn,
		dtype=dtype,seed=seed) 
	else: 
		nn_model=="MPNN"
		neuralNetwork = MessagePassingNeuralNetwork(F,K,cutoff, num_blocks=num_blocks, num_residual_atomic=num_residual_atomic, 
		num_residual_interaction=num_residual_interaction,
		num_residual_output=num_residual_output, num_outputs=num_outputs,
		drop_rate=drop_rate,
		basis_type=basis_type,
		activation_fn=activation_fn,
		output_activation_fn=output_activation_fn,
		dtype=dtype,seed=seed) 

	return neuralNetwork
