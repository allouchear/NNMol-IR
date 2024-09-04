import tensorflow as tf
import os
from Utils.DataContainer import *
from Utils.DataProvider import *
from IRModel.IRModel import *

def getExampleData():
	data = {}
	data['Z'] =  [1,6]
	data['M'] =  [1,6]
	data['Mult'] =  1
	data['Q'] =  0
	data['R'] =  [[0.0,0.0,0.0],[1.0,0.0,0.0]]
	data['idx_i'] =  [0]
	data['idx_j'] =  [1]
	data['mol_idx'] =   tf.zeros_like(data['Z'])
	data['offsets'] =   None
	data['QAlpha'] = 0
	data['QBeta'] = 0
	data['QaAlpha'] = [0,0]
	data['QaBeta'] = [0,0]
	return data


class Evaluator:
	def __init__(self, 
		models_directories,               # directories containing fitted models (can also be a list for ensembles)
		dataFile=os.path.join("Data","B3LYP_NIST_FewCASRemoved.h5"), 
		nvalues=-1,
		batch_size=1, 
		convDistanceToBohr=1.0, 
		dataFrameNameFrequencies="IR_NIST_Freqs_Conv", 
		dataFrameNameIntensities="IR_NIST_Ints_Conv",
		average=False, 
		):



		if(type(models_directories) is not list):
			self._models_directories=[models_directories]
		else:
			self._models_directories=models_directories
		self._average = average



		self._data=DataContainer(dataFile,nameFreqs=dataFrameNameFrequencies, nameInts=dataFrameNameIntensities, convDistanceToBohr=convDistanceToBohr)
		print("data shape = ",self.data.N.shape)
		if nvalues<0:
			ntrain = self.data.N.shape[0]
		else:
			ntrain = nvalues
		print("nvalues=",ntrain)
		self._nvalues=ntrain
		nvalid=0
		ntest=0
		valid_batch_size=0
		self._dataProvider=DataProvider(self.data,ntrain, nvalid, ntest=ntest, batch_size=batch_size,valid_batch_size=valid_batch_size)

		self._models = []
		n=0
		num_frequencies=None
		min_frequencies=None
		max_frequencies=None
		for directory in self._models_directories:
			args = read_model_parameters(directory)
			if self._average:
				average_dir = os.path.join(directory, 'average')
				checkpoint = os.path.join(average_dir, 'average.ckpt')
			else:
				best_dir = os.path.join(directory, 'best')
				checkpoint = os.path.join(best_dir, 'best.ckpt')

			irModel=   IRModel (F=args.num_features,
			K=args.num_basis,
			cutoff=args.cutoff,
			dtype=tf.float64 if args.dtype=='float64' else tf.float32, 
			num_scc=args.num_scc,
			em_type=args.em_type,
			num_hidden_nodes_em=args.num_hidden_nodes_em,
			num_hidden_layers_em=args.num_hidden_layers_em,
			num_blocks=args.num_blocks, 
			num_residual_atomic=args.num_residual_atomic,
			num_residual_interaction=args.num_residual_interaction,
			num_residual_output=args.num_residual_output,
			num_frequencies=args.num_frequencies,
			activation_fn=activation_deserialize(args.activation_function),
			output_activation_fn=activation_deserialize(args.output_activation_function),
			nn_model=args.nn_model,
			basis_type=args.basis_type,
			loss_type=args.loss_type,
			)
			if num_frequencies is None:
				num_frequencies=args.num_frequencies
				max_frequencies=args.max_frequencies
				min_frequencies=args.min_frequencies
			elif num_frequencies != args.num_frequencies:
				print("**********************************************************")
				print("Error number of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			elif abs(max_frequencies-args.max_frequencies)>1e-2:
				print("**********************************************************")
				print("Error max of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			elif abs(min_frequencies-args.min_frequencies)>1e-2:
				print("**********************************************************")
				print("Error min of frequencies are not the same in all models")
				print("**********************************************************")
				sys.exit(1)
			

			data = getExampleData()
			charges, Qa, I, loss, gradients = irModel(data,closs=False) # to set auto shape
			#print("checkpoint=",checkpoint)
			irModel.load_weights(checkpoint)
			self._models.append(irModel)
		self._num_frequencies  = num_frequencies
		self._min_frequencies  = min_frequencies
		self._max_frequencies  = max_frequencies

	def _computeProperties(self, data):
		n=0
		nModel=len(self._models)
		I = None
		molcharges = None
		atomcharges = None
		nhloss =None
		for i in range(nModel):
			lmolcharges, latomcharges, lI, lnhloss = self._models[i].computeProperties(data)
			#print("atomcharges=",atomcharges)

			if i == 0:
				I  = lI
				molcharges = lmolcharges
				atomcharges  = latomcharges
				nhloss = lnhloss
			else:
				n = i+1
				if lI is not None: 
					I = I + (lI-I)/n
				if latomcharges is not None: 
					atomcharges = atomcharges + (latomcharges-atomcharges)/n
				if lmolcharges is not None: 
					molcharges = molcharges+ (lmolcharges-molcharges)/n
				if lnhloss is not None: 
					nhloss = nhloss+ (lnhloss-nhloss)/n 

		return molcharges, atomcharges, I, nhloss

	def computeSums(self, data):
		charges, Qa, I, nhloss = self._computeProperties(data)
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
					d['loss'] =  spectral_loss(I, tf.constant(data[key],dtype=self.model.irModel.dtype),type=self.model.irModel.loss_type)
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
				print("Metrics")
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
		charges, Qa, I, nhloss = self._computeProperties(data)
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
			prefix=os.path.join(metrics_dir,"evaluation")
		elif dataType==1 :
			prefix=os.path.join(metrics_dir,"validation")
		else:
			prefix=os.path.join(metrics_dir,"test")
		if uid is not None:
			prefix=prefix+"_"+str(uid)

		if not os.path.exists(metrics_dir):
			os.makedirs(metrics_dir)

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


	@property
	def data(self):
		return self._data
	@property
	def dataProvider(self):
		return self._dataProvider

	@property
	def models(self):
		return self._models

	@property
	def model(self):
		return self._models[0]
    
	@property
	def nvalues(self):
		return self._nvalues
