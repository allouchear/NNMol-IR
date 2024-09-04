
from math import sqrt
import tensorflow as tf
import numpy as np


class NeighborList:
	"""Neighbor list object.

	cutoffs: list of float
		List of cutoff radii - one for each atom.
	skin: float
		If no atom has moved more than the skin-distance since the
		last call to the ``update()`` method, then the neighbor list
		can be reused.  This will save some expensive rebuilds of
		the list, but extra neighbors outside the cutoff will be
		returned.
	self_interaction: bool
		Should an atom return itself as a neighbor?

	Example::

	  nl = NeighborList([2.3, 1.7])
	  nl.update(R,pbc, cell)
	  indices, offsets = nl.get_neighbors(0)
	  
	"""
	
	def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True):
		self.cutoffs = tf.constant(cutoffs) + skin
		self.skin = skin
		self.sorted = sorted
		self.self_interaction = self_interaction
		self.nupdates = 0
		self.idtype= tf.int32
		self.idtype= tf.int64

	def update(self, R, pbc, cell):
		"""Make sure the list is up to date."""
		if self.nupdates == 0:
		    self.build(R, pbc, cell)
		    return True
		
		if ((self.pbc != pbc).any() or
		    (self.cell != cell).any() or
		    ((self.positions - R)**2).sum(1).max() >
		    self.skin**2):
		    self.build(R,pbc,cell)
		    return True
		
		return False
	
	def build(self, R, pbc, cell):
		"""Build the list."""
		self.positions = R
		self.pbc = pbc
		self.cell = cell
		rcmax = tf.reduce_max(self.cutoffs)
		
		icell = tf.linalg.inv(self.cell)
		scaled = tf.matmul(self.positions, icell)
		scaled0 = scaled

		N = []
		for i in range(3):
		    if self.pbc[i]:
		        scaled0[:, i] %= 1.0
		        v = icell[:, i]
		        h = 1 / sqrt(tf.tensordot(v, v))
		        n =  int(2 * rcmax / h) + 1
		    else:
		        n = 0
		    N.append(n)

		natoms = tf.shape(R)[0]
		Rtype = R.dtype
		offsets = tf.zeros(shape=(natoms,3),dtype=self.idtype)
		#offsets = tf.cast(tf.round(scaled0 - scaled),dtype=self.idtype)
		offsets = tf.dtypes.cast(tf.round(scaled0 - scaled), self.idtype)
		positions0 = tf.matmul(scaled0, self.cell)
		indices = tf.range(natoms,dtype=self.idtype)

		self.nneighbors = 0
		self.npbcneighbors = 0
		self.neighbors = [ [] for a in range(natoms)]
		self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
		for n1 in range(0, N[0] + 1):
			for n2 in range(-N[1], N[1] + 1):
				for n3 in range(-N[2], N[2] + 1):
					if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
						continue
					nn=tf.constant([n1, n2, n3],dtype=self.cell.dtype)
					displacement = tf.tensordot(nn, self.cell,1)
					for a in range(natoms):
						d = positions0 + displacement - positions0[a]
						i = indices[tf.reduce_sum(d**2,axis=1) < (self.cutoffs + self.cutoffs[a])**2]
						if n1 == 0 and n2 == 0 and n3 == 0:
							if self.self_interaction:
								i = i[i >= a]
							else:
								i = i[i > a]
						self.nneighbors += len(i)
						self.neighbors[a] = tf.concat((self.neighbors[a], i),0)
						disp = np.empty((len(i), 3), int)
						disp[:] = (n1,n2,n3)
						for k in range(len(i)):
							disp[k] += offsets[i[k]].numpy() - offsets[a].numpy()

						self.npbcneighbors += disp.any(1).sum()
						self.displacements[a] = np.concatenate((self.displacements[a], disp))

		if self.sorted:
			for a, i in enumerate(self.neighbors):
				mask = (i.numpy() < a)
				if mask.any():
					j = i[mask]
					offsets = self.displacements[a][mask]
					for b, offset in zip(j, offsets.numpy()):
						self.neighbors[b] = tf.concat((self.neighbors[b], [a]))
						self.displacements[b] = tf.concat((self.displacements[b], [-offset]))
					mask = tf.math.logical_not(mask)
					self.neighbors[a] = self.neighbors[a][mask]
					self.displacements[a] = self.displacements[a][mask]
		        
		self.nupdates += 1

	def get_neighbors(self, a):
		"""Return neighbors of atom number a.

		A list of indices and offsets to neighboring atoms is
		returned.  The positions of the neighbor atoms can be
		calculated like this::

		  indices, offsets = nl.get_neighbors(42)
		  for i, offset in zip(indices, offsets):
		      print atoms.positions[i] + dot(offset, atoms.get_cell())

		Notice that if get_neighbors(a) gives atom b as a neighbor,
		then get_neighbors(b) will not return a as a neighbor!"""
		
		return self.neighbors[a], self.displacements[a]

	def get_lists(self, R, pbc, cell):
		"""
		will not return  neighbors_list as 2 arrays"""
		self.update(R,pbc,cell)
		idx_i=[]
		idx_j=[]
		for a, i in enumerate(self.neighbors):
			for j in range(tf.shape(i)[0]):
				idx_i.append(a)
				idx_j.append(self.neighbors[a][j].numpy())
	
		idx_i = tf.dtypes.cast(tf.constant(np.asarray(idx_i)), self.idtype)
		idx_j = tf.constant(np.asarray(idx_j))
		return idx_i, idx_j
