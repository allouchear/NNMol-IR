prefix= ["1s", "2s", "2p", "3s", "3p", "4s", "4p", "3d", "4p", "5s", "4d", "5p", "6s",  "4f", "5d", "6p", "7s", "5f", "5d", "7p" ]

def maxOccl (s):
	if(s[1]) == "s":
		maxc=2
		l=0
	if(s[1]) == "p":
		maxc=6
		l=1
	if(s[1]) == "d":
		maxc=10
		l=2
	if(s[1]) == "f":
		maxc=14
		l=3
	return maxc, l

def conf (atomicNumber):
	Z=atomicNumber
	elecConf=""
	valConf=""
	nmax=0
	lmax=0
	for x in range(len(prefix)):
		maxc,l=maxOccl(prefix[x])
		if Z == 0:
			break
		elif Z >= maxc:
			elecConf = elecConf+prefix[x]+str(maxc)+" "
			if lmax<l:
				lmax=l
			if nmax<int(prefix[x][0]):
				nmax= int(prefix[x][0])
				valConf = " "
			valConf = valConf+prefix[x]+str(maxc)+" "
			Z -= maxc
		elif Z < maxc:
			elecConf = elecConf+prefix[x]+str(Z)
			if lmax<l:
				lmax=l
			if nmax<int(prefix[x][0]):
				nmax= int(prefix[x][0])
				valConf = " "
			valConf = valConf+prefix[x]+str(Z)
			break;
	return elecConf, nmax, lmax, valConf

def valence (atomicNumber):
	Z=atomicNumber
	n=[]
	occ=[]
	l=[]
	nmax=0
	for x in range(len(prefix)):
		maxc,lc=maxOccl(prefix[x])
		nc= int(prefix[x][0])
		if Z == 0:
			break
		elif Z >= maxc:
			if nmax<nc:
				nmax= nc
				n = []
				l = []
				occ = []
			n = n + [nc]
			l = l + [lc]
			occ = occ + [maxc]
			Z -= maxc
		elif Z < maxc:
			if nmax<nc:
				nmax= nc
				n = []
				l = []
				occ = []
			n = n + [nc]
			l = l + [lc]
			occ = occ + [Z]
			break;
	return n,l,occ

def get_l_all_atoms (zmax):
	nl=0
	l = [[-1]]
	for z in range(1,zmax+1):
		nz,lz,occ= valence(z)
		l = l + [lz]
		if nl<len(lz):
			nl = len(lz)
	return l,nl
