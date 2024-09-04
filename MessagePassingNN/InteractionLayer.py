import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from .ResidualLayer import *

class InteractionLayer(Layer):
	def __str__(self):
		return "interaction_layer"+super().__str__()

	def __init__(self, F, num_residual, activation_fn=None, seed=None, drop_rate=0.0, dtype=tf.float32, name="InteractionLayer"):
		super().__init__(dtype=dtype, name=name)

		self._drop_rate = drop_rate
		self._activation_fn = activation_fn


		initializer = tf.keras.initializers.GlorotNormal(seed=seed)


		#transforms radial basis functions to feature space
		self._k2f = layers.Dense(F, activation=None, kernel_initializer='zeros', bias_initializer='zeros', 
				use_bias=False, dtype=dtype, name=name+"/k2f")
		#rearrange feature vectors for computing the "message"
		self._dense_i = layers.Dense(F, activation=activation_fn, kernel_initializer=initializer, bias_initializer='zeros', 
				use_bias=True, dtype=dtype, name=name+"/densei") # central atoms
		self._dense_j = layers.Dense(F, activation=activation_fn, kernel_initializer=initializer, bias_initializer='zeros', 
				use_bias=True, dtype=dtype, name=name+"/densej")  # neighbouring atoms
		#for performing residual transformation on the "message"
		self._residual_layer = []
		#print("num_residual=",num_residual)
		for i in range(num_residual):
			self._residual_layer.append(ResidualLayer(F, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype,name=name+"/Residual"+str(i)))

		#for performing the final update to the feature vectors
		self._dense = layers.Dense(F, activation=None, kernel_initializer=initializer, bias_initializer='zeros', 
				use_bias=True, dtype=dtype, name=name+"/denseFinal")
		self._u = tf.Variable(tf.ones([F], dtype=dtype), name=name+"/u", dtype=dtype,trainable=True)
		tf.summary.histogram("gates",  self.u)  

	@property
	def activation_fn(self):
		return self._activation_fn

	@property
	def drop_rate(self):
		return self._drop_rate

	@property
	def k2f(self):
		return self._k2f

	@property
	def dense_i(self):
		return self._dense_i

	@property
	def dense_j(self):
		return self._dense_j

	@property
	def residual_layer(self):
		return self._residual_layer

	@property
	def dense(self):
		return self._dense

	@property
	def u(self):
		return self._u
    
	def __call__(self, x, rbf, idx_i, idx_j):
		#pre-activation
		if self.activation_fn is not None: 
			xa = tf.nn.dropout(self.activation_fn(x), self.drop_rate)
		else:
			xa = tf.nn.dropout(x, self.drop_rate)
		#calculate feature mask from radial basis functions
		g = self.k2f(rbf)
		#print("g=",g)
		#calculate contribution of neighbors and central atom
		xi = self.dense_i(xa)
		xj = tf.math.segment_sum(g*tf.gather(self.dense_j(xa), idx_j), idx_i)
		#add contributions to get the "message" 
		m = xi + xj 
		for i in range(len(self.residual_layer)):
			m = self.residual_layer[i](m)
		if self.activation_fn is not None: 
			m = self.activation_fn(m)
		x = self.u*x + self.dense(m)
		return x
