"""
Data from h5 using DataFrame
"""

import sys
import numpy  as np
import pandas as pd
from .PeriodicTable import *

def getCID(df):
	return np.array(df.index)

def getN(df):
	if "Atoms" in df.columns.values:
		atoms=df["Atoms"].values
		NAtoms=[ len(listA) for listA in atoms]
		NAtoms=np.array(NAtoms)
		return NAtoms
	else:
		return None

def getAtomicCharges(df):
	Q=None
	if "Partial Charges (NPA)" in df.columns.values:
		Q=df["Partial Charges (NPA)"].values
	elif "Partial Charges (ESP)" in df.columns.values:
		Q=df["Partial Charges (ESP)"].values
	elif "Partial Charges (Hirshfeld)" in df.columns.values:
		Q=df["Partial Charges (Hirshfeld)"].values
	elif "Partial Charges (Mulliken)" in df.columns.values:
		Q=df["Partial Charges (Mulliken)"].values
	if np.isnan(Q).any():
		return None
	else:
		return Q

def getByName(df,name):
	V=None
	if name in df.columns.values:
		V=df[name].values
	else:
		#print("columns=",df.columns.values)
		print(name, " is not in columns of the dataframe")

	return V

"""
check if the length is the same for all molecules
return this value is the same
"""
def getFreqsIntsLength(V,type="frequencies"):
	nMols=len(V)
	if nMols<1:
		sys.stderr.write("=============================\n")
		sys.stderr.write("Error : Number of molecules=0\n")
		sys.stderr.write("=============================\n")
		sys.exit(1)

	nlength=len(V[0])
	for v in V:
		if nlength != len(v):
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("Error : Number of {0:s} is not the same for all molecules\n".format(type))
			sys.stderr.write("Error : You must do a convolution to obtain the same number of {0:s} for all molecules\n".format(type))
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)

	return nlength

"""
check if the max f is the same for all molecules
return this value is the same
"""
def getFreqsMax(V):
	nMols=len(V)
	if nMols<1:
		sys.stderr.write("=============================\n")
		sys.stderr.write("Error : Number of molecules=0\n")
		sys.stderr.write("=============================\n")
		sys.exit(1)

	maxf=max(V[0])
	for v in V:
		if abs(maxf - max(v))>1e-2:
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("Error : The max frequency is not the same for all molecules\n")
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)

	return maxf

"""
check if the min f is the same for all molecules
return this value is the same
"""
def getFreqsMin(V):
	nMols=len(V)
	if nMols<1:
		sys.stderr.write("=============================\n")
		sys.stderr.write("Error : Number of molecules=0\n")
		sys.stderr.write("=============================\n")
		sys.exit(1)

	minf=min(V[0])
	for v in V:
		if abs(minf - min(v))>1e-2:
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("Error : The min frequency is not the same for all molecules\n")
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)

	return minf

class DataContainer:
	def __repr__(self):
		return "DataContainer"
	def __init__(self, filename, nameFreqs="frequenciesConv", nameInts="irIntensitiesConv",convDistanceToBohr=1.0):
		#read in data
		print("nameFreqs=",nameFreqs)
		print("nameInts=",nameInts)

		df = pd.read_hdf(filename)
		#print(df.index)
		print("=================== Database info ==================================================================")
		print("Number of molecules=",df.shape[0])
		#print("Data info\n",df.info())
		print("columns=",df.columns.values)

		#Frequencies
		self._F = getByName(df,nameFreqs)
		if self._F is None:
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("I cannot read IR frequencies from {0:s} file\n".format(filename))
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)
		print("Len frequencies=",len(self.F[0]))
		self._lenF = getFreqsIntsLength(self.F)
		self._maxF = getFreqsMax(self.F)
		self._minF = getFreqsMin(self.F)
		#Intensitis
		self._I = getByName(df,nameInts)
		if self._I is None:
			sys.stderr.write("I cannot read IR intensities from {0:s} file\n".format(filename))
			sys.exit(1)

		if(self.lenF != getFreqsIntsLength(self.I)):
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("Error : The length of IR intensities != length of IR Frequencies in {0:s} file\n".format(filename))
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)

		#positions (cartesian coordinates)
		self._R = getByName(df,name="Coordinates")
		if self._R is not None:	 
			self._R = self._R*convDistanceToBohr 
		else:
			sys.stderr.write("=============================================================================================\n")
			sys.stderr.write("I cannot read IR frequencies from {0:s} file\n".format(filename))
			sys.stderr.write("=============================================================================================\n")
			sys.exit(1)

		#print("shapeR=",self.R.shape)
		#print("shapeI=",self.I.shape)
		#print("shapeF=",self.F.shape)

		#number of atoms
		self._N = getN(df)

		# Mol index
		self._CID = getCID(df)
		print("CID=",self.CID)

		# masses
		self._M = getByName(df,name="Masses")
		self._M =(np.array(self.M)/1822.88848121).tolist() # in amu to set masses near Z

		#atomic numbers/nuclear charges
		self._Z = getByName(df,name="Atoms")
		if self.Z is not None: 
			if self.M is None:
				periodicTable=PeriodicTable()
				self._M = [] 
				for zMol in self.Z:
					mMol = []
					for z in zMol:
						if int(z)==0:
							mMol.append(0)
						else:
							mass=periodicTable.elementZ(int(z)).isotopes[0].rMass # mass of first istotope
							mMol.append(mass)
					self._M.append(mMol)
				self._M = np.array(self._M)
		else:
			self._Z = None

		# Molecular charges
		self._Q=getByName(df, name="Charge")

		#reference atomic charges
		self._Qa = getAtomicCharges(df)
		#print("Qa=", self._Qa)

		#maximum number of atoms per molecule
		#print("Z=", self.Z)
		nZmax=0
		for zMol in self.Z:
			if nZmax<len(zMol):
				nZmax = len(zMol)
		#print("nZmax=", nZmax)
		self._N_max	= nZmax
		#print("Nmax=",self._N_max)

		# Spin multiplicity (2S+1 not S)
		self._Mult=getByName(df, name="Multiplicity")
		if self._Mult is None: 
			self._Mult = np.ones(self.N.shape[0],dtype=float)
			# if number of electrons is odd , set multiplicity to 2
			if self.Z is not None and self.Q is not None:
				for im, zMol in enumerate(self.Z):
					ne = np.sum(zMol)
					ne = ne - self.Q[im]
					ne = int(ne+0.5)
					if ne%2==1:
						self._Mult[im] = 2

		#reference total charge by spin alpha, beta
		self._QAlpha = getByName(df,name="QAlpha")
		if self._QAlpha is None:
			self._QAlpha = 0.5*(self.Q-self.Mult+1)

		self._QBeta = getByName(df,name="QBeta")
		if self._QBeta is None:
			self._QBeta = 0.5*(self.Q+self.Mult-1)

		print("Number of atoms by molecule =", self.N)
		print("Number of atoms for smallest molecule =", np.array(self.N).min())
		print("Number of atoms for largest molecule  =", np.array(self.N).max())

		#reference Alpha atomic charges
		self._QaAlpha  = getByName(df,name="QaAlpha")
		if self._QaAlpha is None:
			self._QaAlpha = []
			for im in range(self.N.shape[0]):
				m = []
				N = self.N[im]
				m = [self._QAlpha[im]/N]*N
				nres =  self.N_max-N
				if nres>0:
					m = m + [0.0]*nres
				self._QaAlpha.append(m)
		self._QaAlpha = np.asarray(self.QaAlpha)

		#reference Beta atomic charges
		self._QaBeta  = getByName(df,name="QaBeta")
		if self._QaBeta is None:
			self._QaBeta = []
			for im in range(self.N.shape[0]):
				m = []
				N = self.N[im]
				m = [self._QBeta[im]/N]*N
				nres =  self.N_max-N
				if nres>0:
					m = m + [0.0]*nres
				self._QaBeta.append(m)
		self._QaBeta = np.asarray(self.QaBeta)

	
		#construct indices used to extract position vectors to calculate relative positions 
		#(basically, constructs indices for calculating all possible interactions (excluding self interactions), 
		#this is a naive (but simple) O(N^2) approach, could be replaced by something more sophisticated) 
		self._idx_i = np.empty([self.N_max, self.N_max-1],dtype=int)
		for i in range(self.idx_i.shape[0]):
			for j in range(self.idx_i.shape[1]):
				self._idx_i[i,j] = i

		self._idx_j = np.empty([self.N_max, self.N_max-1],dtype=int)
		for i in range(self.idx_j.shape[0]):
			c = 0
			for j in range(self.idx_j.shape[0]):
				if j != i:
					self._idx_j[i,c] = j
					c += 1
		print("====================================================================================================")

	@property
	def N_max(self):
		return self._N_max

	@property
	def N(self):
		return self._N

	@property
	def CID(self):
		return self._CID

	@property
	def Z(self):
		return self._Z

	@property
	def M(self):
		return self._M

	@property
	def Q(self):
		return self._Q

	@property
	def Qa(self):
		return self._Qa

	@property
	def Mult(self):
		return self._Mult

	@property
	def QAlpha(self):
		return self._QAlpha

	@property
	def QBeta(self):
		return self._QBeta

	@property
	def QaAlpha(self):
		return self._QaAlpha

	@property
	def QaBeta(self):
		return self._QaBeta

	@property
	def I(self):
		return self._I

	@property
	def R(self):
		return self._R

	@property
	def F(self):
		return self._F

	@property
	def lenF(self):
		return self._lenF

	@property
	def maxF(self):
		return self._maxF

	@property
	def minF(self):
		return self._minF

	#indices for atoms i (when calculating interactions)
	@property
	def idx_i(self):
		return self._idx_i

	#indices for atoms j (when calculating interactions)
	@property
	def idx_j(self):
		return self._idx_j

	def __len__(self): 
		return self.Z.shape[0]

	def __getitem__(self, idx):
		if type(idx) is int or type(idx) is np.int64:
			idx = [idx]

		data = {	'I':		 [],
				'F':		 [],
				'Z':		 [],
				'M':		 [],
				'Q':		 [],
				'Qa':	 	 [],
				'Mult':	 	 [],
				'CID':	 	 [],
				'QAlpha': 	 [],
				'QBeta': 	 [],
				'QaAlpha': 	 [],
				'QaBeta': 	 [],
				'R':		 [],
				'idx_i':	 [],
				'idx_j':	 [],
				'mol_idx': [],
				'offsets'  : []
			}

		Ntot = 0 #total number of atoms
		for k, i in enumerate(idx):
			N = self.N[i] #number of atoms
			I = N*(N-1)   #number of interactions
			#append data
			if self.CID is not None:
				data['CID'].append(self.CID[i])
			else:
				data['CID'].append(np.nan)
			if self.Q is not None:
				data['Q'].append(self.Q[i])
			else:
				data['Q'].append(np.nan)
			if self.Mult is not None:
				data['Mult'].append(self.Q[i])
			else:
				data['Mult'].append(np.nan)
			if self.QAlpha is not None:
				data['QAlpha'].append(self.QAlpha[i])
			else:
				data['QAlpha'].append(np.nan)
			if self.QBeta is not None:
				data['QBeta'].append(self.QBeta[i])
			else:
				data['QBeta'].append(np.nan)

			if self.Qa is not None:
				data['Qa'].extend(self.Qa[i,:N].tolist())
			else:
				data['Qa'].extend([np.nan]*N)

			if self.QaAlpha is not None:
				data['QaAlpha'].extend(self.QaAlpha[i,:N].tolist())
			else:
				data['QaAlpha'].extend([np.nan]*N)

			if self.QaBeta is not None:
				data['QaBeta'].extend(self.QaBeta[i,:N].tolist())
			else:
				data['QaBeta'].extend([np.nan]*N)

			if self.Z is not None:
				#data['Z'].extend(self.Z[i,:N].tolist())
				data['Z'].extend(self.Z[i])
			else:
				data['Z'].append(0)

			if self.M is not None:
				#data['M'].extend(self.M[i,:N].tolist())
				data['M'].extend(self.M[i])
			else:
				data['M'].append(6.0)

			if self.R is not None:
				#data['R'].extend(self.R[i,:N,:].tolist())
				data['R'].extend(self.R[i].reshape(-1,3).tolist())
			else:
				data['R'].extend([[np.nan,np.nan,np.nan]])

			if self.F is not None:
				#data['F'].extend(self.F[i,:].tolist())
				data['F'].extend([self.F[i]])
			else:
				data['F'].extend([[np.nan]*1000])
			if self.I is not None:
				#data['I'].extend(self.I[i,:].tolist())
				data['I'].extend([self.I[i]])
			else:
				data['I'].extend([[np.nan]*1000])

			data['idx_i'].extend(np.reshape(self.idx_i[:N,:N-1]+Ntot,[-1]).tolist())
			data['idx_j'].extend(np.reshape(self.idx_j[:N,:N-1]+Ntot,[-1]).tolist())
			#offsets could be added in case they are need
			data['mol_idx'].extend([k] * N)
			#increment totals
			Ntot += N
		#print("I=",np.array(data['I']).shape)
		#print("Z=",np.array(data['Z']).shape)
		#print("M=",np.array(data['M']).shape)
		#print("R=",np.array(data['R']).shape)

		return data

