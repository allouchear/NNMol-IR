import tensorflow as tf
from tensorflow.keras import layers

class ResidualLayer(layers.Layer):
	def __str__(self):
		return "residual_layer"+super().__str__()

	def __init__(self, n_out, activation_fn=None, use_bias=True, seed=None, drop_rate=0.0, dtype=tf.float32,name="residual_layer"):
		super().__init__(dtype=dtype,name=name)
		"""
			n_in is defined in input x, in __call__
		"""
		if dtype==tf.float64 :
			tf.keras.backend.set_floatx('float64')

		self._drop_rate = drop_rate
		self._activation_fn = activation_fn
		initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		self._dense = layers.Dense(n_out, activation=activation_fn, 
				kernel_initializer=initializer, bias_initializer='zeros',name=name+"/Dense",
				use_bias=use_bias, dtype=dtype)
		self._residual = layers.Dense(n_out, activation=None,
				kernel_initializer=initializer, bias_initializer='zeros', name=name+"/Residual",
				use_bias=use_bias, dtype=dtype)
      
	@property
	def activation_fn(self):
		return self._activation_fn

	@property
	def drop_rate(self):
		return self._drop_rate

	@property
	def dense(self):
		return self._dense

	@property
	def residual(self):
		return self._residual

	def __call__(self, x):
		#pre-activation
		if self.activation_fn is not None: 
			y = tf.nn.dropout(self.activation_fn(x), self.drop_rate)
		else:
			y = tf.nn.dropout(x, self.drop_rate)
		x += self.residual(self.dense(y))
		return x
