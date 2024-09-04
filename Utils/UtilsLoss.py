#from __future__ import absolute_import
import tensorflow as tf
import numpy  as np


THRFREQ=1e-8

def spectral_sid_loss(model_spectra ,target_spectra,threshold=THRFREQ):
	nan_mask=tf.math.logical_or(tf.math.is_nan(target_spectra) , tf.math.is_nan(model_spectra))
	model_spectra = tf.where(model_spectra<threshold, threshold, model_spectra)
	target_spectra = tf.where(target_spectra<threshold, threshold, target_spectra)
	zero_sub=tf.zeros_like(target_spectra)
	sum_model_spectra = tf.reduce_sum(tf.where(nan_mask,zero_sub,model_spectra),axis=1)
	sum_model_spectra = tf.expand_dims(sum_model_spectra,1)
	loss = tf.ones_like(target_spectra)
	model_spectra = tf.where(nan_mask, 1.0, model_spectra)
	target_spectra = tf.where(nan_mask, 1.0, target_spectra)
	loss = tf.math.log(model_spectra/target_spectra)*model_spectra+\
		tf.math.log(target_spectra/model_spectra)*target_spectra
	loss = tf.where(nan_mask, 0.0, loss)
	loss=tf.reduce_sum(loss)
	return loss

def spectral_sae_loss(model_spectra ,target_spectra,threshold=THRFREQ):
	loss = tf.reduce_sum(tf.abs(model_spectra-target_spectra))
	return loss

def spectral_loss(model_spectra ,target_spectra,threshold=THRFREQ, type='SID'):
	if type.upper()=='SID':
		loss=  spectral_sid_loss(model_spectra ,target_spectra,threshold=threshold)
	else:
		loss=  spectral_sae_loss(model_spectra ,target_spectra,threshold=threshold)
	return loss
