import tensorflow as tf
import os
import sys
import re
from .IRicalConstants import *
from .NeighborList import *

def scitodeci(sci):
        tmp=re.search(r'(\d+\.?\d+)\*\^(-?\d+)',sci)
        return float(tmp.group(1))*pow(10,float(tmp.group(2)))

class Molecule:
	""" Provides a general purpose molecule"""
	#def __init__(self, Z =  tf.zeros(shape=(1,),dtype=tf.uint32), R = tf.zeros(shape=(1,3),dtype=tf.float32)):
	def __init__(self, Z, R, cell=[[1.0,0.0,0.0], [0.0,1.0,0.0],[0.0,0.0,1.0],], pbc=[False,False,False], cutoffs=None):
		"""
		Args:
			Z: tf(shape=(natoms),dtype=tf.uint32) of atomic numbers.
			R: tf(shape=(natoms,3),dtype=tf.float32) atomic coordinates in bohr
		"""
		self.Z = Z
		self.R = R
		self.cell = tf.constant(cell, dtype=R.dtype,name="cell")
		self.pbc  = tf.constant(pbc, dtype=tf.bool, name="pbc")
		self.cutoffs = None
		if cutoffs!=None:
			self.cutoffs = tf.constant(cutoffs, dtype=R.dtype, name="cutoffs")
		print("Zmol=",Z)
		print("Rmol=",R)
		self.properties = {}
		self.name=None

		self.neighborList = None
		self.idx_i = None 
		self.idx_j = None
		if cutoffs!=None:
			buildNeighborList(self)
		return

	def buildNeighborList(self,cutoffs=None):
		"""
		build neighbor list
		"""
		if self.cutoffs==None and cutoffs!=None:
			self.cutoffs = tf.constant(cutoffs, dtype=self.R.dtype, name="cutoffs")

		if self.cutoffs==None:
			natoms=self.Z.shape[0]
			self.cutoffs = tf.ones([natoms], dtype=self.R.dtype, name="cutoffs")
			self.cutoffs *= 1e10 # big cutoff

		if self.cutoffs.shape[0] != self.Z.shape[0]:
			natoms=self.Z.shape[0]
			oneval=self.cutoffs[0]
			self.cutoffs = tf.ones([natoms], dtype=self.R.dtype, name="cutoffs")
			self.cutoffs *= oneval # same value for all atoms
		self.neighborList = NeighborList(self.cutoffs,sorted=False,self_interaction=False)
		self.idx_i, self.idx_j = self.neighborList.get_lists(self.R,self.pbc,self.cell)
		return
	def getNeighborList(self,cutoffs=None):
		if self.idx_i==None:
			self.buildNeighborList(cutoffs)

		return self.idx_i, self.idx_j

	def buildElectronicConfiguration(self,charge=0,spin=1):
		"""
		fill up electronic configuration.
		"""
		nelectron = tf.reduce_sum(self.Z) - charge
		nalpha = (nelectron+spin)//2
		nbeta = nalpha - self.spin
		basis = []
		self.properties["basis"] = basis
		self.properties["charge"] = charge
		self.properties["spin"] = spin
		self.properties["nalpha"] = nalpha
		self.properties["nbeta"] = nbeta
		return

	def numberOfElectrons(self):
		return tf.reduce_sum(self.Z)

	def isIsomer(self,other):
		return (tf.reduce_all(tf.equal(tf.sort(self.Z), tf.sort(other.Z))))

	def nAtoms(self):
		return self.Z.shape[0]

	def numberOfAtoms(self, e):
		return tf.reduce_sum( [1 if at==e else 0 for at in self.atoms ] )

	def read_xyz_with_properties(self, path, properties, center=True):
		try:
			f=open(path,"r")
			lines=f.readlines()
			natoms=int(lines[0])
			Ztype = self.Z.dtype
			Rtype = self.R.dtype
			self.Z = tf.zeros(shape=(natoms,),dtype=Ztype)
			self.R = tf.zeros(shape=(natoms,3),dtype=Rtype)
			for i in range(natoms):
				line = lines[i+2].split()
				self.atoms[i]=AtomicNumber(line[0])
				for j in range(3):
					try:
						self.R[i,j]=Rtype(line[j+1])
					except:
						self.R[i,j]=scitodeci(line[j+1])
			if center:
				self.R -= self.center()
			properties_line = lines[1]
			for i, mol_property in enumerate(properties):
				if mol_property == "name":
					self.properties["name"] = properties_line.split(";")[i]
				if mol_property == "energy":
					self.properties["energy"] = float(properties_line.split(";")[i])
					self.CalculateAtomization()
				if mol_property == "gradients":
					self.properties['gradients'] = tf.zeros((natoms, 3))
					read_forces = (properties_line.split(";")[i]).split(",")
					for j in range(natoms):
						for k in range(3):
							self.properties['gradients'][j,k] = float(read_forces[j*3+k])
				if mol_property == "dipole":
					self.properties['dipole'] = tf.zeros((3))
					read_dipoles = (properties_line.split(";")[i]).split(",")
					for j in range(3):
						self.properties['dipole'][j] = float(read_dipoles[j])
				if mol_property == "partial_charges":
					self.properties["partial_charges"] = tf.zeros((natoms))
					read_charges = (properties_line.split(";")[i]).split(",")
					for j in range(natoms):
						self.properties["partial_charges"] = float(read_charges[j])
			f.close()
		except Exception as Ex:
			print("Read Failed.", Ex)
			raise Ex
		return

	def propertyString(self):
		tore = ""
		for prop in self.properties.keys():
			try:
				if (prop == "energy"):
					tore = tore +";;;"+prop+" "+str(self.properties["energy"])
				elif (prop == "Lattice"):
					tore = tore +";;;"+prop+" "+(self.properties[prop]).tostring()
				else:
					tore = tore +";;;"+prop+" "+str(self.properties[prop])
			except Exception as Ex:
				# print "Problem with energy", string
				pass
		return tore

	def __str__(self,wprop=False):
		lines =""
		natom = self.Z.shape[0]
		if (wprop):
			lines = lines+(str(natom)+"\nComment: "+self.propertyString()+"\n")
		else:
			lines = lines+(str(natom)+"\nComment: \n")
		for i in range (natom):
			atom_name =  list(atoi.keys())[list(atoi.values()).index(self.Z[i])]
			if (i<natom-1):
				lines = lines+(atom_name+"   "+str(self.R[i][0].numpy())+ "  "+str(self.R[i][1].numpy())+ "  "+str(self.R[i][2].numpy())+"\n")
			else:
				lines = lines+(atom_name+"   "+str(self.R[i][0].numpy())+ "  "+str(self.R[i][1].numpy())+ "  "+str(self.R[i][2].numpy()))
		return lines

	def __repr__(self):
		return self.__str__()

	def writeXYZfile(self, fpath=".", fname="mol", mode="a", wprop = False):
		if not os.path.exists(os.path.dirname(fpath+"/"+fname+".xyz")):
			try:
				os.makedirs(os.path.dirname(fpath+"/"+fname+".xyz"))
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
		with open(fpath+"/"+fname+".xyz", mode) as f:
			for line in self.__str__(wprop).split("\n"):
				f.write(line+"\n")

	def center(self, momentOrder = 1.):
		''' Returns the center of atom or mass

		Args:
			momentOrder: Option to do nth order moment.
		Returns:
			Center of Atom, or a higher-order moment.
		'''
		return tf.reduce_mean(tf.power(self.R,momentOrder),axis=0)

	def rms(self, m):
		""" Cartesian coordinate difference. """
		err  = 0.0
		for i in range (0, (self.R).shape[0]):
			err += tf.linalg.norm(m.R[i] - self.R[i])
		return err/self.R.shape[0]
