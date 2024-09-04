import tensorflow as tf
from tensorflow.keras.layers import Layer
from .InteractionLayer import *
from .ResidualLayer    import *

class InteractionBlock(Layer):
	def __str__(self):
		return "interaction_block"+super().__str__()

	def __init__(self, F, num_residual_atomic, num_residual_interaction, activation_fn=None, seed=None, drop_rate=0.0, dtype=tf.float32, name="InteractionBlock"):
		super().__init__(dtype=dtype,name=name)
            #interaction layer
		self._interaction = InteractionLayer(F, num_residual_interaction, activation_fn=activation_fn, name=name+'/InteractionLayer',
				    seed=seed, drop_rate=drop_rate, dtype=dtype)

		#residual layers
		self._residual_layer = []
		for i in range(num_residual_atomic):
			self._residual_layer.append(ResidualLayer(F, activation_fn=activation_fn, seed=seed, drop_rate=drop_rate, dtype=dtype,name=name+"/Residual"+str(i)))

	@property
	def interaction(self):
		return self._interaction
    
	@property
	def residual_layer(self):
		return self._residual_layer

	def __call__(self, x, rbf, idx_i, idx_j):
		x = self.interaction(x, rbf, idx_i, idx_j)
		for i in range(len(self.residual_layer)):
			x = self.residual_layer[i](x)
		return x
