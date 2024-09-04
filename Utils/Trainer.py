import tensorflow as tf
import tensorflow_addons as tfa
import os
from Utils.DataContainer import *
from Utils.DataProvider import *
from IRModel.IRModel import *
from Utils.UtilsLoss import *
import sys

class ReduceLROnPlateau:
	def __str__(self):
		return "Min loss=" + str(self.loss.numpy()) + ", Wait=" + str(self.wait) + ", Patience=" + str(self.patience) + ", Min_lr=" + str(self.min_lr) + ", Factor=" + str(self.factor)

	def __init__(self, patience=5, factor=0.1, min_lr=0):
		self._patience = patience
		self._factor = factor
		self._min_lr = min_lr
		self._wait = 0
		self._loss = None

	def update(self,loss):
		if self.loss is None:
			self._loss=loss
			return False

		if self.loss>loss:
			self._loss=loss
			self._wait = 0
			return False

		if self.wait<self.patience:
			self._wait += 1
			return False
		else:
			self._wait = 0
			return True

	@property
	def patience(self):
		return self._patience

	@property
	def factor(self):
		return self._factor

	@property
	def min_lr(self):
		return self._min_lr

	@property
	def wait(self):
		return self._wait

	@property
	def loss(self):
		return self._loss

def print_optimizer(args):
	nsteps=args.max_steps
	print("================================== Optimizer ========================================================")
	if args.learning_schedule_type.upper()=="EXP" :
		print("Schedules exponentialDecay : initial_learning_rate * decay_rate**(numsteps / decay_steps) ")
		print("                           = {:f} * {:f}**(numstep / {:d}) ".format(args.learning_rate, args.decay_rate, args.decay_steps))
		print("     initial learning rate     : ", args.learning_rate)
		print("     last  learning rate       : ", args.learning_rate * args.decay_rate**(nsteps / args.decay_steps))

	elif args.learning_schedule_type.upper()=="TIME" :
		print("Schedules InverseTimeDecay :  initial_learning_rate / (1 + decay_rate * step / decay_step)")
		print("                           = {:f} / ( 1+ {:f}*numstep / {:d})".format(args.learning_rate, args.decay_rate, args.decay_steps))
		print("     initial learning rate     : ", args.learning_rate)
		print("     last  learning rate       : ", args.learning_rate /(1+args.decay_rate*nsteps / args.decay_steps))
	elif args.learning_schedule_type.upper()=="PLATEAU" :
		print("Plateau learning :")
		print("------------------")
		print("      Initial leraning rate         :  ", args.learning_rate)
		print("      Patience                      :  ", args.patience)
		print("      Factor                        :  ", args.decay_rate)
		print("      Min_lr                        :  ", args.min_lr)
		plateau = ReduceLROnPlateau(patience=args.patience, factor=args.decay_rate, min_lr=args.min_lr)
	else:
		print("Constant leraning rate         :  ", args.learning_rate)

	print("Optimizer:")
	print("---------:")
	print("      Adam, beta_1=0.9, beta_2=", args.ema_decay, "epsilon=1e-10,amsgrad=True") 
	if args.use_average>0:
		print("      With MovingAverage")
	print("====================================================================================================")

def get_optimizer(args):
	plateau = None
	nsteps=args.max_steps
	if args.learning_schedule_type.upper()=="EXP" :
		learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=args.decay_steps, decay_rate=args.decay_rate)

	elif args.learning_schedule_type.upper()=="TIME" :
		learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=args.learning_rate, decay_steps=args.decay_steps, decay_rate=args.decay_rate,staircase=False)
	elif args.learning_schedule_type.upper()=="PLATEAU" :
		learning_rate = args.learning_rate
		plateau = ReduceLROnPlateau(patience=args.patience, factor=args.decay_rate, min_lr=args.min_lr)
	else:
		learning_rate = args.learning_rate

	#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=args.ema_decay, epsilon=1e-10,amsgrad=True)
	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=args.ema_decay, epsilon=1e-10,amsgrad=True)
	if args.use_average>0:
		optimizer = tfa.optimizers.MovingAverage(optimizer)
	return  optimizer,plateau

class Trainer:
	def __init__(self, model, args, dataFile=os.path.join("Data","data.h5")):
		self._args = args
		self._optimizer,self._plateau = get_optimizer(args)
		print_optimizer(args)
		self._use_average = args.use_average
		self._restart_optimizer_nsteps = args.restart_optimizer_nsteps
		self._model = model
		self._istep = 1
		self._data=DataContainer(dataFile,nameFreqs=args.dataFrameNameFrequencies, nameInts=args.dataFrameNameIntensities, convDistanceToBohr=args.conv_distance_to_bohr)
		if self.model.num_outputs-1 != self.lenF:
			sys.stderr.write("==============================================================================================================\n")
			sys.stderr.write("Error : Number of frequencies/intensities in DataFile is not the same in num_frequencies defined in input file\n")
			sys.stderr.write("      : Number of frequencies/intensities in DataFile = {0:s} \n".format(str(self.lenF)))
			sys.stderr.write("      : Number of input file = {0:s} \n".format(str(self.model.num_outputs-1)))
			sys.stderr.write("==============================================================================================================\n")
			sys.exit(1)

		self._dataProvider=DataProvider(self.data,args.num_train, args.num_valid, ntest=args.num_test, batch_size=args.batch_size,valid_batch_size=args.valid_batch_size,seed=args.seed)
		self._dataTrain=self.dataProvider.get_all_train_data()

		#fxyz="data.xyz"
		#save_xyz(self._dataTrain, fxyz)
		#print("XYZ data train saved in file ", fxyz)

		#print("Model=",model)
		charges, Qa, I, loss, gradients = self.model(self.dataProvider.next_batch()) # to set auto shape 
		self._bck = self.model.get_weights()
		#print("learning_rateInit= ", self.optimizer.lr(0).numpy())
		self._loss_type = args.loss_type
		self._learning_schedule_type = args.learning_schedule_type 
		self._verbose = args.verbose

	def applyOneStep(self,  dt=None, verbose=True):
		if self.restart_optimizer_nsteps > 0 and (self._istep)%self.restart_optimizer_nsteps==0:
				self.reset_optimizer()
		if dt is None:
			dt = self.dataProvider.next_batch()
		charges, Qa, I, loss, gradients = self.model(dt)
		if verbose is True:
			print("Current learning rate = ",self.get_learning_rate())
			print("Loss=",  loss.numpy())
			print("I=",  I.numpy().tolist())
			print("Q=",  charges.numpy().tolist())
			print("Qref     =",  dt['Q'])
			print("Qa=",  Qa.numpy().tolist())
			print("Qaref     =",  dt['Qa'])
			#print("gradients=",  gradients)
			#print("gradients size=",  [g.shape for g in gradients])
			#print(dt)
			print_gradients_norms(gradients,self.model.trainable_weights,details=False)
			#print("==========================================================")
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
		self.update_learning_rate(loss)
			
		self._istep += 1
		return loss, gradients

	def computeSums(self, data):
		charges, Qa, I, nhloss = self.model.computeProperties(data)
		sums= {}
		values = { 'I':I, 'Q':charges, 'Qa':Qa, 'hloss':nhloss}
		coefs = { 'I':1.0, 'Q':0, 'Qa':0, 'hloss':self.model.irModel.nhlambda}
		for key in coefs:
			if coefs[key] > 0:
				datakey=tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1])
				predkey=tf.reshape(values[key],[-1])
				nan_mask=tf.math.logical_or(tf.math.is_nan(predkey) , tf.math.is_nan(datakey))
				datakey = tf.where(nan_mask, 0.0, datakey)
				predkey = tf.where(nan_mask, 0.0, predkey)
				sData=tf.reduce_sum(datakey)
				sPredict=tf.reduce_sum(predkey)
				s2Data=tf.reduce_sum(tf.square(datakey))
				s2Predict=tf.reduce_sum(tf.square(predkey))
				prod=datakey*predkey
				sDataPredict=tf.reduce_sum(prod)
				s2DataPredict=tf.reduce_sum(tf.square(prod))
				sdif=tf.math.abs(predkey-datakey)
				sAbsDataMPredict=tf.reduce_sum(sdif)
				"""
				sData=tf.reduce_sum(tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1]))
				sPredict=tf.reduce_sum(tf.reshape(values[key],[-1]))
				s2Data=tf.reduce_sum(tf.square(tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1])))
				s2Predict=tf.reduce_sum(tf.square(tf.reshape(values[key],[-1])))
				sDataPredict=tf.reduce_sum((tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1]))*tf.reshape(values[key],[-1]))
				s2DataPredict=tf.reduce_sum(tf.square((tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1]))*tf.reshape(values[key],[-1])))
				sAbsDataMPredict=tf.reduce_sum(tf.math.abs((tf.reshape(tf.constant(data[key],dtype=self.model.irModel.dtype),[-1]))-tf.reshape(values[key],[-1])))
				"""
				d = {}
				d['sData']         =  sData
				d['sPredict']      =  sPredict
				d['s2Data']        =  s2Data
				d['s2Predict']     =  s2Predict
				d['sDataPredict']  =  sDataPredict
				d['s2DataPredict'] =  s2DataPredict
				d['sAbsDataMPredict'] = sAbsDataMPredict
				#d['n'] =  tf.reshape(values[key],[-1]).shape[0]
				d['n'] = datakey.shape[0]
				if key=='I':
					d['loss'] =  spectral_loss(I, tf.constant(data[key],dtype=self.model.irModel.dtype),type=self.loss_type)
					#print("nI=",d['n'],'loss_type=', self.loss_type, 'dloss=',d['loss'],"lossDiv=",d['loss']/d['n'])
				elif key=='hloss':
					d['loss'] =  nhloss*d['n']
				else:
					d['loss'] =  sAbsDataMPredict
					#print("nOther=",d['n'])
				sums[key] = d

		return sums

	def addSums(self, data, sums=None):
		coefs = { 'I':1.0, 'Q':0, 'Qa':0, 'hloss':self.model.irModel.nhlambda}
		if sums is None:
			sums= {}
			lis=[ 'sData', 'sPredict', 's2Data', 's2Predict', 'sDataPredict', 's2DataPredict', 'sAbsDataMPredict','loss','n']
			for key in coefs:
				if coefs[key] > 0:
					d = {}
					for name in lis:
						d[name]  =  0
					sums[key] = d
		s=self.computeSums(data)
		for keys in sums:
			for key in sums[keys]:
				sums[keys][key] +=  s[keys][key]
		return sums

	def addTrainSums(self, sums=None):
		sums=self.addSums(self.dataProvider.current_batch(), sums=sums)
		return sums


	def computeAccuraciesFromSums(self, sums, verbose=False,dataType=0):
		coefs = { 'I':1.0, 'Q':0, 'Qa':0, 'hloss':self.model.irModel.nhlambda}
		acc = {}
		mae = {}
		ase = {}
		rmse = {}
		R2 = {}
		rr = {}
		lossV = {}
		for keys in sums:
			mae[keys] =  sums[keys]['sAbsDataMPredict']/sums[keys]['n']
			lossV[keys] =  sums[keys]['loss']/sums[keys]['n']
			m =  sums[keys]['sData']/sums[keys]['n']
			residual =  sums[keys]['s2Data'] +sums[keys]['s2Predict'] -2*sums[keys]['sDataPredict']
			if tf.math.abs(residual)<1e-14:
				R2[keys] = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				total =  sums[keys]['s2Data']-2*m*sums[keys]['sData']+sums[keys]['n']*m*m
				R2[keys] = 1.0-residual/(total+1e-14)

			ymean = m
			ypredmean = sums[keys]['sPredict']/sums[keys]['n']
			ase[keys] = ypredmean-ymean 
			rmse[keys] = tf.math.sqrt(tf.math.abs(residual/sums[keys]['n']))

			yvar =  sums[keys]['s2Data']-2*ymean*sums[keys]['sData']+sums[keys]['n']*ymean*ymean
			ypredvar =  sums[keys]['s2Predict']-2*ypredmean*sums[keys]['sPredict']+sums[keys]['n']*ypredmean*ypredmean
			cov =  sums[keys]['sDataPredict']-ypredmean*sums[keys]['sData']-ymean*sums[keys]['sPredict']+sums[keys]['n']*ymean*ypredmean
			den = tf.sqrt(yvar*ypredvar)
			if den<1e-14:
				corr = tf.Variable(1.0,dtype=sums[keys]['sData'].dtype)
			else:
				corr = cov/tf.sqrt(yvar*ypredvar+1e-14)
			rr[keys] = corr*corr

		loss=0.0
		for key in coefs:
			#if mae[key] in locals()  and coefs[key]>0:
			if coefs[key]>0:
				loss += lossV[key]*coefs[key]

		lossDic ={ 'L':loss}
		acc['mae'] = mae
		acc['ase'] = ase
		acc['rmse'] = rmse
		acc['R2'] = R2
		acc['r2'] = rr
		acc['Loss'] = lossDic
		if verbose is True:
			if dataType==0:
				print("Train metrics")
			elif dataType==1:
				print("Validation metrics")
			else:
				print("Test metrics")
			print("--------------------------------")
			for keys in acc:
				#print(keys,":")
				#[ print(keys,"[", key,"]=", acc[keys][key].numpy()) for key in acc[keys] ]
				[ print("{:5s}[{:2s}] = {:20.10f}".format(keys,key,acc[keys][key].numpy())) for key in acc[keys] ]
				#print("")
		return acc

	def computeLossFromSums(self, sums):
		coefs = { 'I':1.0}
		lossV = {}
		for keys in sums:
			lossV[keys] =  sums[keys]['loss']/sums[keys]['n']

		loss=0.0
		for key in coefs:
			if coefs[key]>0:
				loss += lossV[key]*coefs[key]

		return loss.numpy()

	def computeAccuracies(self, verbose=True, dataType=0):
		sums= None
		if dataType==0:
			nsteps = self.dataProvider.get_nsteps_batch()
		elif dataType==1:
			nsteps = self.dataProvider.get_nsteps_valid_batch()
		else:
			nsteps = self.dataProvider.get_nsteps_test_batch()

		if dataType==0:
			self.dataProvider.set_train_idx_to_end() # needed for train idx beaucuse we shuffle train part after each epoch

		for i in range(nsteps):
			if dataType==0:
				dt = self.dataProvider.next_batch()
			elif dataType==1:
				dt = self.dataProvider.next_valid_batch()
			else:
				dt = self.dataProvider.next_test_batch()
			sums=self.addSums(dt, sums)
		if nsteps>0:
			acc = self.computeAccuraciesFromSums(sums,verbose,dataType=dataType)
		else: 
			acc= None
		return acc

	def computeTrainAccuracies(self, verbose=True):
		return self.computeAccuracies(verbose=verbose, dataType=0)

	def computeValidationAccuracies(self, verbose=True):
		return self.computeAccuracies(verbose=verbose, dataType=1)

	def computeTestAccuracies(self, verbose=True):
		return self.computeAccuracies(verbose=verbose, dataType=2)

	"""
		implemeneted only to test self.computeAccuracies
	"""
	def computeAccuraciesTry(self,  dt=None, verbose=True, dataType=-1):
		if dt is None:
			if dataType==0 :
				dt = self.dataProvider.get_all_train_data()
			elif dataType==1:
				dt = self.dataProvider.get_all_valid_data()
			elif dataType==2:
				dt = self.dataProvider.get_all_test_data()
			else:
				dt = self.dataProvider.next_batch()

		acc=self.model.computeAccuracies(dt)
		if verbose is True:
			for keys in acc:
				#print(keys,":")
				[ print(keys,"[", key,"]=", acc[keys][key].numpy()) for key in acc[keys] ]
				print("")

		return acc

	def computeAccuracy(self,  dt=None, verbose=True):
		if dt is None:
			dt = self.dataProvider.next_batch()
		means,loss=self.model.computeAccuracy(dt)
		if verbose is True:
			[ print("Mean[", key,"]=", means[key].numpy()) for key in means ]
			print("Loss",loss)

		return means,loss

	def computeTrainAccuracy(self, verbose=True):
		acc = self.computeTrainAccuracies(verbose=verbose)
		if acc is not None:
			means = acc['mae']
			loss = acc['Loss']['L']
		else:
			means = None
			loss = None
		return means,loss

	def computeValidationAccuracy(self, verbose=True):
		acc = self.computeValidationAccuracies(verbose=verbose)
		if acc is not None:
			means = acc['mae']
			loss = acc['Loss']['L']
		else:
			means = None
			loss = None
		return means,loss

	"""
		Save reference and predicted values in text file, as 2 columns
	"""
	def saveAnalysisOneData(self, ref, pred, fileOut, CID=None):
		cidexpended=None
		if CID is not None and len(CID)==len(ref):
			cidexpended=[]
			for k in range(len(CID)):
				cidexpended += [CID[k]]*len(ref[k])

		rref = tf.reshape(tf.convert_to_tensor(ref,dtype=self.model.irModel.dtype),[-1])
		rpred = tf.reshape(tf.convert_to_tensor(pred,dtype=self.model.irModel.dtype),[-1])
		for i in range(rref.shape[0]):
			st = '\n'
			if cidexpended is not None:
				st = " " +'{:30s}\n'.format(cidexpended[i])
			fileOut.write(" "+'{:20.14e}'.format(rref.numpy()[i])+" "+'{:20.14e}'.format(rpred.numpy()[i])+st)

	def saveAnalysisData(self, data, files):
		charges, Qa, I, nhloss = self.model.computeProperties(data)
		coefs = { 'I':1.0}
		values = { 'I':I, 'Q':charges, 'Qa':Qa}
		for key in coefs:
			if coefs[key] > 0:
				self.saveAnalysisOneData(data[key], values[key], files[key], CID=data['CID'])

	def addTitleAnalysis(self, fileOut, t):
		fileOut.write("###########################################################################\n")
		fileOut.write("#"+'{:20s}'.format(t)+"\n")
		fileOut.write("###########################################################################\n")

	def saveAnalysis(self, metrics_dir, dataType=0, uid=None):
		if dataType==0:
			prefix=os.path.join(metrics_dir,"train")
		elif dataType==1 :
			prefix=os.path.join(metrics_dir,"validation")
		else:
			prefix=os.path.join(metrics_dir,"test")
		if uid is not None:
			prefix=prefix+"_"+str(uid)

		fileNames= {}
		titles = {}
		fileNames['I'] = prefix+"_spectra.txt"
		s="I"
		#titles['I'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s)
		titles['I'] ="#"+'{:20s}'.format("Reference "+s)+" "+'{:20s}'.format("Predicted "+s) +" "+'{:30s}'.format("CID")

		files= {}
		for key in fileNames:
			files[key] = open(fileNames[key],"w")
			self.addTitleAnalysis(files[key], titles[key])

		if dataType==0:
			nsteps = self.dataProvider.get_nsteps_batch()
		elif dataType==1:
			nsteps = self.dataProvider.get_nsteps_valid_batch()
		else:
			nsteps = self.dataProvider.get_nsteps_test_batch()

		if dataType==0:
			self.dataProvider.set_train_idx_to_end() # needed for train idx beaucuse we shuffle train part after each epoch

		for i in range(nsteps):
			if dataType==0:
				dt = self.dataProvider.next_batch()
			elif dataType==1:
				dt = self.dataProvider.next_valid_batch()
			else:
				dt = self.dataProvider.next_test_batch()
			#save_xyz(dt, "Train_"+id_generator()+".xyz")
			self.saveAnalysisData(dt, files)

		for key in files:
			files[key].close()

		return fileNames


	def load_weights(self, fname):
		ok=False
		checkpoint_dir = os.path.dirname(fname)
		ffname=os.path.join(checkpoint_dir,"checkpoint")
		if tf.io.gfile.exists(ffname):
			self.model.load_weights(fname)
			ok=True

		if not ok:
			print("Warrning : I cannot read ",fname, "file")
		return ok

	def save_weights(self, fname):
		self.model.save_weights(fname)

	def save_averaged_weights(self, fname):
		self.save_variable_backups()
		self.optimizer.assign_average_vars(self.model.variables)
		self.model.save_weights(fname)
		self.restore_variable_backups()

	def save_variable_backups(self):
		self._bck = self.model.get_weights()

	def restore_variable_backups(self):
		self.model.set_weights(self.bck)

	def set_average_vars(self):
		self.save_variable_backups()
		self._optimizer.assign_average_vars(self.model.variables)

	def get_learning_rate(self):
		if isinstance(self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
			current_lr = self.optimizer.lr(self.optimizer.iterations)
		else:
			current_lr = self.optimizer.lr
		return current_lr.numpy()

	def set_learning_rate(self, new_learning_rate):
		if isinstance(self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
			sys.stderr.write("????????????????????????????????????????????????????????????????????\n")
			sys.stderr.write("Warning, the learning rate is now constant. Schedules is deactivate\n")
			sys.stderr.write("????????????????????????????????????????????????????????????????????\n")
			self._optimizer.lr=tf.constant(new_learning_rate)
		else:
			self._optimizer.lr=tf.constant(new_learning_rate)

	def reset_optimizer(self):
		if self.verbose>=1:
			print("?????????????????????????????????????")
			print("Warning : We restart the optimizer")
			print("?????????????????????????????????????")
		self._optimizer,self._plateau = get_optimizer(self.args)

	def update_learning_rate(self, loss):
		if self.learning_schedule_type.upper()=="PLATEAU":
			if self._plateau.update(loss) is True:
				newlr= self._optimizer.lr * self.plateau.factor
				if newlr>= self.plateau.min_lr:
					self.set_learning_rate(newlr)
			if self.verbose>=1:
				print("After Plateau update:")
				print("    Current lr        : ",self.get_learning_rate())
				print("    ReduceLROnPlateau : ",str(self.plateau))

	@property
	def bck(self):
		return self._bck

	@property
	def data(self):
		return self._data

	@property
	def dataTrain(self):
		return self._dataTrain

	@property
	def dataProvider(self):
		return self._dataProvider

	@property
	def optimizer(self):
		return self._optimizer

	@property
	def lenF(self):
		return self._data.lenF

	@property
	def maxF(self):
		return self._data.maxF

	@property
	def minF(self):
		return self._data.minF

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def use_average(self):
		return self._use_average

	@property
	def learning_schedule_type(self):
		return self._learning_schedule_type

	@property
	def plateau(self):
		return self._plateau

	@property
	def verbose(self):
		return self._verbose

	@property
	def model(self):
		return self._model
    
	@property
	def istep(self):
		return self._istep

	@property
	def restart_optimizer_nsteps(self):
		return self._restart_optimizer_nsteps
       
	@property
	def args(self):
		return self._args
       
