import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.layers import Layer

#inverse softplus transformation
def softplus_inverse(x):
    return x + tf.math.log(-tf.math.expm1(-x))

#radial basis function expansion
class RBFLayer(Layer):
	def __str__(self):
		#return "radial_basis_function_layer"+super().__str__()
		return  "Radial basis type                   : " +str(self.basis_type)\
			+"\n"+\
			"Number of radial basis functions    : " + str(self.K)\
			+"\n"+\
			"Cutoff distance                     : " + str(self.cutoff.numpy())

	def __init__(self, K, cutoff, name=None,  beta=0.2, basis_type="Default", dtype=tf.float32):
		super().__init__(dtype=dtype)
		self._K = K
		self._cutoff = cutoff
		if basis_type=="Default":
			basis_type="GaussianNet"
		#basis_type="Bessel"
		#basis_type="Slater"
		#basis_type="Gaussian"
		self._basis_type = basis_type
		if self.basis_type=="Gaussian":
			centers = tf.cast(tf.linspace(tf.constant(0.0,dtype=dtype),cutoff,K),dtype=dtype)
			self._centers = tf.Variable(centers, name=name+"centers", dtype=dtype, trainable=False)
			tf.summary.histogram("rbf_centers", self.centers) 
			delta_rs =  tf.constant(cutoff,dtype=self.dtype)/(K-1)
			alpha = beta/(delta_rs**2)
			widths = [alpha]*K
			self._widths = tf.Variable(widths,  name=name+"widths",  dtype=dtype, trainable=False)
			tf.summary.histogram("rbf_widths", self.widths)
		elif self.basis_type=="GaussianNet":
			centers = softplus_inverse(tf.linspace(tf.constant(1.0,dtype=dtype),tf.math.exp(-cutoff),K))
			self._centers = tf.nn.softplus(tf.Variable(centers, name=name+"centers", dtype=dtype, trainable=False))
			tf.summary.histogram("rbf_centers", self.centers) 
			#initialize widths (inverse softplus transformation is applied, such that softplus can be used to guarantee positive values)
			widths = [softplus_inverse((0.5/((1.0-tf.math.exp(-cutoff))/K))**2)]*K
			self._widths = tf.nn.softplus(tf.Variable(widths,  name=name+"widths",  dtype=dtype, trainable=False))
			tf.summary.histogram("rbf_widths", self.widths)

		elif self.basis_type=="Slater": # centers at origin 
			K=int(tf.sqrt(K*1.0).numpy())
			alphas=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			alphas=alphas.numpy().tolist()*K
			n=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			n=tf.repeat(n,repeats=K)
			ccut=tf.cast(cutoff,dtype=dtype)
			self._alphas = tf.Variable(alphas, name=name+"alphas", dtype=dtype, trainable=False) # r**(n-1)*exp(-alphas*(rij/rc))
			self._n = tf.Variable(n, name=name+"n", dtype=dtype, trainable=False) 
			K=K*K
			self._K=K
		else: # bessel, radial = sin ( Johannes Klicpera et al., https://arxiv.org/pdf/2003.03123.pdf)
			n=tf.linspace(tf.constant(1.0,dtype=dtype),K,K)
			print("n=",n)
			ccut=tf.cast(cutoff,dtype=dtype)
			alphas = n*math.pi/ccut # n pi/cutoff => normc*sin(alpha*rij)/rij
			self._alphas = tf.Variable(alphas, name=name+"alphas", dtype=dtype, trainable=False)
			normc=tf.math.sqrt(2.0/ccut)
			self._normc = tf.Variable(normc, name=name+"normc", dtype=dtype, trainable=False)

	@property
	def K(self):
		return self._K

	@property
	def cutoff(self):
		return self._cutoff
    
	@property
	def centers(self):
		return self._centers   

	@property
	def widths(self):
		return self._widths  

	@property
	def coefs(self):
		return self._coefs

	@property
	def alphas(self):
		return self._alphas

	@property
	def n(self):
		return self._n

	@property
	def normc(self):
		return self._normc

	@property
	def basis_type(self):
		return self._basis_type


	#cutoff function that ensures a smooth cutoff
	def cutoff_fn(self, rij):
		x = rij/self.cutoff
		x3 = x**3
		x4 = x3*x
		x5 = x4*x
		return tf.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, tf.zeros_like(x))

	def cutoff_fncos(self, rij):
		x = rij/self.cutoff
		return tf.where(x < 1, (0.5*(1.0+tf.math.cos(math.pi*x)))**2, tf.zeros_like(x))

	def radial(self, rij):
		if self.basis_type=="Gaussian":
			v = tf.exp(-self.widths*(rij-self.centers)**2) # Gaussian
			#v = tf.exp(-self.widths*tf.sqrt((rij-self.centers)**2)) # Slater
			v *= self.cutoff_fncos(rij)
			return v;
		elif self.basis_type=="GaussianNet":
			v = tf.exp(-self.widths*(tf.exp(-rij)-self.centers)**2)
			v *= self.cutoff_fn(rij)
			return v;
		elif self.basis_type=="Slater":
			x = rij/self.cutoff
			v = x**self.n*tf.exp(-self.alphas*x) # Gaussian
			v *= self.cutoff_fncos(rij)
			return v;
		else:
			v = self.normc*tf.math.sin(self.alphas*rij)/rij #  From Bessel
			v *= self.cutoff_fncos(rij)
			return v;
    
    
	def __call__(self, rij):
		rij = tf.expand_dims(rij, -1)
		rbf = self.radial(rij)
		return rbf



