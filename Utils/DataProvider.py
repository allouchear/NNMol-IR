"""
Tensorflow implementation of IRNet
"""
import tensorflow as tf
import numpy  as np

class DataProvider:
	def __repr__(self):
		return "DataProvider"
	def __init__(self, data, ntrain, nvalid, ntest=-1, batch_size=1, valid_batch_size=1, seed=None):
		self._data = data
		self._ndata  = len(data)
		self._ntrain = ntrain
		self._nvalid = nvalid
		self._ntest  = len(data)-self.ntrain-self.nvalid
		if ntest>=0 and ntest<self._ntest:
			self._ntest  = ntest

		self._batch_size = batch_size
		self._valid_batch_size = valid_batch_size
		self._test_batch_size  = valid_batch_size
		if batch_size>ntrain :
			self._batch_size = ntrain
		if valid_batch_size>nvalid :
			self._valid_batch_size = nvalid
		if self.test_batch_size>self.ntest :
			self._test_batch_size = self.ntest

		#random state parameter, such that random operations are reproducible if wanted
		self._random_state = np.random.RandomState(seed=seed)

		#create shuffled list of indices
		idx = self._random_state.permutation(np.arange(len(self.data)))

		#idx = np.arange(len(self.data)) # TO BE REMOVE, it is for test
		#print("idxDataProvider=",idx)

		#store indices of training, validation and test data
		self._idx_train = idx[0:self.ntrain]
		self._idx_valid = idx[self.ntrain:self.ntrain+self.nvalid]
		self._idx_test  = idx[self.ntrain+self.nvalid:]

		#for retrieving batches
		self._idx_in_epoch = 0 
		self._valid_idx = 0
		self._test_idx = 0

		self._idx_in_epoch_old = 0 

		# number_of_batches = len([_ for _ in iter(self._dataset_train)])

	@property
	def data(self):
		return self._data

	@property
	def ndata(self):
		return self._ndata

	@property
	def ntrain(self):
		return self._ntrain
	
	@property
	def nvalid(self):
		return self._nvalid
	
	@property
	def ntest(self):
		return self._ntest

	@property
	def random_state(self):
		return self._random_state

	@property
	def idx_train(self):
		return self._idx_train

	@property
	def idx_valid(self):
		return self._idx_valid   

	@property
	def idx_test(self):
		return self._idx_test

	@property
	def idx_in_epoch(self):
		return self._idx_in_epoch

	@property
	def idx_in_epoch_old(self):
		return self._idx_in_epoch_old


	@property
	def valid_idx(self):
		return self._valid_idx

	@property
	def test_idx(self):
		return self._test_idx


	@property
	def batch_size(self):
		return self._batch_size

	@property
	def valid_batch_size(self):
		return self._valid_batch_size

	@property
	def test_batch_size(self):
		return self._test_batch_size

	#shuffle the training data
	def shuffle(self):
		self._idx_train = self.random_state.permutation(self.idx_train)

	#returns a batch of samples from the training set
	def next_batch(self):
		start = self.idx_in_epoch
		self._idx_in_epoch += self.batch_size
		#epoch is finished, set needs to be shuffled
		if start >= self.ntrain:
			self.shuffle()
			start = 0
			self._idx_in_epoch = self.batch_size
		# case where ntrain is not a multiple of batch_size
		if self.idx_in_epoch > self.ntrain:
			self._idx_in_epoch = self.ntrain

		end = self.idx_in_epoch   
		self._idx_in_epoch_old = start
		return self.data[self.idx_train[start:end]]

	#returns a batch of samples from the validation set
	def next_valid_batch(self):
		start = self.valid_idx
		self._valid_idx += self.valid_batch_size
		#finished one pass-through, reset index
		if start >= self.nvalid:
			start = 0
			self._valid_idx = self.valid_batch_size

		# case where nvalid is not a multiple of valid_batch_size
		if self.valid_idx > self.nvalid:
			self._valid_idx = self.nvalid

		end =  self.valid_idx
		return self.data[self.idx_valid[start:end]]

	#returns a batch of samples from the test set
	def next_test_batch(self):
		start = self.test_idx
		self._test_idx += self.test_batch_size
		#finished one pass-through, reset index
		if start >= self.ntest:
			start = 0
			self._test_idx = self.test_batch_size

		# case where ntest is not a multiple of test_batch_size
		if self.test_idx > self.ntest:
			self._test_idx = self.ntest

		end =  self.test_idx
		return self.data[self.idx_test[start:end]]

	#returns the current batch of samples from the training set
	def current_batch(self):
		start = self.idx_in_epoch_old
		end = self.idx_in_epoch   
		return self.data[self.idx_train[start:end]]

	def set_train_idx_to_end(self):
		self._idx_in_epoch=self.ntrain


	def reset_batch(self):
		start = 0
		self._idx_in_epoch = self.batch_size

	def reset_valid_batch(self):
		start = 0
		self._valid_idx = self.valid_batch_size

	def rest_test_batch(self):
		start = 0
		self._test_idx = self.test_batch_size

	def get_nsteps_batch(self):
		nsteps = 0
		if self.batch_size != 0:
			nsteps=self.ntrain//self.batch_size
			if self.ntrain%self.batch_size != 0:
				nsteps += 1
		return nsteps

	def get_nsteps_valid_batch(self):
		nsteps = 0
		if self.valid_batch_size != 0:
			nsteps=self.nvalid//self.valid_batch_size
			if self.nvalid%self.valid_batch_size != 0:
				nsteps += 1
		return nsteps

	def get_nsteps_test_batch(self):
		nsteps = 0
		if self.test_batch_size !=0:
			nsteps=self.ntest//self.test_batch_size
			if self.ntest%self.test_batch_size !=0:
				nsteps += 1
		return nsteps

	def create_tensorboard_histograms(self, directory):
		w = tf.summary.create_file_writer(directory)
		with w.as_default():
			dataTrain=self.data[self.idx_train]
			dataValid=self.data[self.idx_valid]
			dataTest=self.data[self.idx_test]
			for key in dataTrain:
				if not (key=='mol_idx' or key=='idx_i' or  key=='idx_j' or  key=='offsets' or 'CID'):
					tf.summary.histogram(key+"_train", dataTrain[key], step=0)
					tf.summary.histogram(key+"_valid", dataValid[key], step=0)
					tf.summary.histogram(key+"_test", dataTest[key], step=0)
			print("Histogram of data created in ", directory)
			print("To visualise them : tensorboard --logdir ", directory)
			print(" and run your navigator to show all ")


	def get_data(self, idx):
		return self.data[idx]

	def get_train_data(self, i):
		idx = self.idx_train[i]
		return self.data[idx]

	def get_all_train_data(self):
		return self.data[self.idx_train]

	def get_valid_data(self, i):
		idx = self.idx_valid[i]
		return self.data[idx]

	def get_all_valid_data(self):
		return self.data[self.idx_valid]

	def get_test_data(self, i):
		idx = self.idx_test[i]
		return self.data[idx]
	
	def get_all_test_data(self):
		return self.data[self.idx_test]
