import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers

class ElementalModesBlock(Layer):
	def __str__(self):
		return "output"+super().__str__()

	def __init__(self, F, num_hidden_nodes=None, num_hidden_layers=2, activation_fn=None, seed=None, drop_rate=0.0, use_bias=True, dtype=tf.float32, name="ElementalModesBlock"):
		super().__init__(dtype=dtype,name=name)

		self._activation_fn = activation_fn

		if dtype==tf.float64 :
			tf.keras.backend.set_floatx('float64')

		if num_hidden_nodes is None :
			num_hidden_nodes = F

		initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		self._hidden_layers = []
		#self._batch_normalizations = []
		for i in range(num_hidden_layers):
			self._hidden_layers.append(
				layers.Dense(num_hidden_nodes, activation=activation_fn, kernel_initializer=initializer, 
				bias_initializer='zeros',name=name+"/Hidden", use_bias=use_bias, dtype=dtype
				))
			#self._batch_normalizations.append(layers.BatchNormalization(trainable=True, dtype=dtype))

		initializer = tf.keras.initializers.GlorotNormal(seed=seed)
		#initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed)
		self._latent_features = layers.Dense(F, activation=None, 
				kernel_initializer=initializer, bias_initializer='zeros',
				#kernel_initializer='zeros', bias_initializer='zeros',
				use_bias=True, dtype=dtype,name=name+'/latent_features')

	@property
	def activation_fn(self):
		return self._activation_fn
    
	@property
	def hidden_layers(self):
		return self._hidden_layers
	#@property
	#def batch_normalizations(self):
	#	return self._batch_normalizations

	@property
	def latent_features(self):
		return self._latent_features

	def __call__(self, x):
		for i in range(len(self.hidden_layers)):
			x = self.hidden_layers[i](x)
			#x = self.batch_normalizations[i](x)
		return self.latent_features(x)
