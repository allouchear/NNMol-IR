import tensorflow as tf

#google's swish function
def swish(x):
	return x*tf.sigmoid(x)

#First time softplus was used as activation function: "Incorporating Second-Order Functional Knowledge for Better Option Pricing"
#(https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf)
def _softplus(x):
	return tf.math.log1p(tf.math.exp(x))

def softplus(x):
    #this definition is for numerical stability for x larger than 15 (single precision) 
    #or x larger than 34 (double precision), there is no numerical difference anymore 
    #between the softplus and a linear function
	return tf.where(x < 15.0, _softplus(tf.where(x < 15.0, x, tf.zeros_like(x))), x) 

def shifted_softplus(x):
	return tf.nn.softplus(x) - tf.math.log(tf.cast(2.0, x.dtype))

#this ensures that the function is close to linear near the origin!
def scaled_shifted_softplus(x):
	return 2*shifted_softplus(x)

#is not really self-normalizing sadly...
def self_normalizing_shifted_softplus(x):
	return 1.875596256135042*shifted_softplus(x)

#general: log((exp(alpha)-1)*exp(x)+1)-alpha
def smooth_ELU(x):
	return tf.math.log1p(1.718281828459045*tf.math.exp(x))-1.0 #(e-1) = 1.718281828459045

def self_normalizing_smooth_ELU(x):
	return 1.574030675714671*smooth_ELU(x)

def self_normalizing_asinh(x): 
	return 1.256734802399369*tf.asinh(x)

def self_normalizing_tanh(x): 
	return 1.592537419722831*tf.tanh(x)

def self_abs(x): 
	return tf.math.abs(x)

def smooth_abs3(x,k=3.0): 
	return  2.0/k*tf.math.log(1.0+tf.math.exp(k*x))-x-2.0/k*tf.math.log(2.0)

def smooth_abs(x): 
	#return  0.66666666666666666*tf.math.log(1.0+tf.math.exp(3.0*x))-x-0.66666666666666666*tf.math.log(tf.constant(2,dtype=x.dtype))
	k=tf.constant(3,dtype=x.dtype)
	return  2/k*tf.math.log(1+tf.math.exp(k*x))-x-2/k*0.69314718055994528623

def activation_deserialize(name):
	if name=="None":
		return None
	elif name=="softplus":
		return softplus
	elif name=="shifted_softplus":
		return shifted_softplus 
	elif name=="scaled_shifted_softplus":
		return scaled_shifted_softplus
	elif name=="self_normalizing_shifted_softplus":
		return self_normalizing_shifted_softplus
	elif name=="smooth_ELU":
		return smooth_ELU
	elif name=="self_normalizing_smooth_ELU":
		return self_normalizing_smooth_ELU
	elif name=="self_normalizing_asinh":
		return  self_normalizing_asinh
	elif name=="self_normalizing_tanh":
		return  self_normalizing_tanh
	elif name=="self_abs":
		return  self_abs
	elif name=="smooth_abs":
		return  smooth_abs
	else:
		return tf.keras.activations.deserialize(name)
