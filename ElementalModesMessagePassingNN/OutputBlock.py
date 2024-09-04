import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from .ResidualLayer import *

class OutputBlock(Layer):
	def __str__(self):
		return "output"+super().__str__()

	def __init__(self, F, n_out, num_residual, activation_fn=None, seed=None, drop_rate=0.0, dtype=tf.float32, name="OutputBlock"):
		super().__init__(dtype=dtype,name=name)

		self._activation_fn = activation_fn

		if dtype==tf.float64 :
			tf.keras.backend.set_floatx('float64')

		self._residual_layer = []
		for i in range(num_residual):
			self._residual_layer.append(ResidualLayer(F, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype,name=name+"/Residual"+str(i)))

		#initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)
		self._dense = layers.Dense(n_out, activation=None, 
				kernel_initializer=initializer, bias_initializer='zeros',
				#kernel_initializer='zeros', bias_initializer='zeros',
				#use_bias=False, dtype=dtype,name=name+'/Dense')
				use_bias=True, dtype=dtype,name=name+'/Dense')

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def residual_layer(self):
		return self._residual_layer

	@property
	def dense(self):
		return self._dense

	def __call__(self, x):
		for i in range(len(self.residual_layer)):
			x = self.residual_layer[i](x)
		if self.activation_fn is not None: 
			x = self.activation_fn(x)
		return self.dense(x)
