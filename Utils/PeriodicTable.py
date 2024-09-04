"""
Tensorflow implementation of IRNet (see https://arxiv.org/abs/1902.08408)
obtained from: https://github.com/MMunibas/IRNet
"""

import numpy  as np

BOHR_TO_ANG = 0.52917726
HARTREE_TO_EV  = 27.21138505

# All values are in atomic units 

class Isotope:
	def __repr__(self):
		return "Isotope"
	def __str__(self):
		return "("+str(self._iMass)+"," +str(self._rMass)+","+str(self._abundance)+")"
	def __init__(self, iMass, rMass, abundance):
		self._iMass=iMass
		self._rMass=rMass
		self._abundance=abundance
	@property
	def iMass(self):
		return self._iMass
	@property
	def rMass(self):
		return self._rMass
	@property
	def abundance(self):
		return self._abundance

class Element:
	def __repr__(self):
		return "Element"
	def __str__(self):
		st="name="+str(self._name)+" "+"symbol="+str(self._symbol)+" "+"atomicNumber="+str(self._atomicNumber)+" "+"covalentRadii="+str(self._covalentRadii)+" "+"bondOrderRadii="+str(self._bondOrderRadii)+" "+"vanDerWaalsRadii="+str(self._vanDerWaalsRadii)+" "+"radii="+str(self._radii)+" "+"maximumBondValence="+str(self._maximumBondValence)+" "+"mass="+str(self._mass)+" "+"electronegativity="+str(self._electronegativity)+" "+"red="+str(self._red)+" "+"green="+str(self._green)+" "+"blue="+str(self._blue)+" "
		for i in range(len(self._isotopes)):
			if i==0:
				st += "Isotopes :"
			st += self._isotopes[i].__str__()+" ; "
		return st

	def __init__(self, name, symbol, atomicNumber, covalentRadii, bondOrderRadii, vanDerWaalsRadii, radii, maximumBondValence,
		mass, electronegativity, red, green, blue):
		self._name=name
		self._symbol=symbol
		self._atomicNumber=atomicNumber
		self._covalentRadii=covalentRadii
		self._bondOrderRadii=bondOrderRadii
		self._vanDerWaalsRadii=vanDerWaalsRadii
		self._radii=radii
		self._maximumBondValence=maximumBondValence
		self._mass=mass
		self._electronegativity=electronegativity
		self._red=red
		self._green=green
		self._blue=blue
		self._isotopes=[]
	@property
	def name(self):
		return self._name
	@property
	def symbol(self):
		return self._symbol
	@property
	def atomicNumber(self):
		return self._atomicNumber
	@property
	def covalentRadii(self):
		return self._covalentRadii
	@property
	def bondOrderRadii(self):
		return self._bondOrderRadii
	@property
	def vanDerWaalsRadii(self):
		return self._vanDerWaalsRadii
	@property
	def radii(self):
		return self._radii
	@property
	def maximumBondValence(self):
		return self._maximumBondValence
	@property
	def mass(self):
		return self._mass
	@property
	def electronegativity(self):
		return self._electronegativity
	@property
	def red(self):
		return self._red
	@property
	def green(self):
		return self._green
	@property
	def blue(self):
		return self._blue

	@property
	def isotopes(self):
		return self._isotopes

class PeriodicTable:
	def __repr__(self):
		return "PeriodicTable"
	def __init__(self):
		self._elements=[]
		self._add_all_elements()
		#print(self._elements[0].name)
		#print("len=",len(self._elements))
		self._add_all_isotopes()

	def _add_element(self, name, symbol, atomicNumber, covalentRadii, bondOrderRadii, vanDerWaalsRadii, radii, maximumBondValence,
		mass, electronegativity, red, green, blue):
		element=Element(name, symbol, atomicNumber, covalentRadii, bondOrderRadii, vanDerWaalsRadii, radii, maximumBondValence,
		mass, electronegativity, red, green, blue)
		self._elements.append(element)

	def _add_isotope(self, symbol, iMass, rMass, abundance):
		for i in range(len(self._elements)):
			if self._elements[i].symbol==symbol :
				self._elements[i]._isotopes.append(Isotope(iMass, rMass, abundance))
				return
	@property
	def elements(self):
		return self._elements

	def element(self,symbol):
		for e in self._elements:
			if e.symbol==symbol:
				return e
		return None

	def elementZ(self,z):
		for e in self._elements:
			if e.atomicNumber==z:
				return e
		return None

	def getCovalentRadius(self):
		# elements already sorted by atomic number (Z)
		values=[0.0] # firt is Dummy atom
		for e in self._elements:
				values.append(e.covalentRadii)
		return values

	def _add_all_elements(self):
		self._add_element("Hydrogen", "H", 1, 0.812582, 0.623610, 2.267671, 0.453534, 1, 1.007825, 0.077178, 1, 1, 1 )
		self._add_element("Helium", "He", 2, 2.135390, 1.322808, 2.645616, 0.529123, 0, 4.002603, 0.000000, 0.849989, 1, 1 )
		self._add_element("Lithium", "Li", 3, 1.662959, 2.324363, 3.439301, 0.687860, 1, 7.016004, 0.036016, 0.8, 0.499992, 1 )
		self._add_element("Beryllium", "Be", 4, 1.039349, 1.700753, 3.212534, 0.642507, 2, 9.012182, 0.057699, 0.759991, 1, 0 )
		self._add_element("Boron", "B", 5, 1.946418, 1.549575, 3.930630, 0.786126, 3, 11.009305, 0.074972, 1, 0.709987, 0.709987 )
		self._add_element("Carbon", "C", 6, 1.662959, 1.455089, 3.684966, 0.736993, 4, 12.000000, 0.093716, 0.499992, 0.499992, 0.499992 )
		self._add_element("Nitrogen", "N", 7, 1.662959, 1.322808, 3.495993, 0.699199, 3, 14.003074, 0.111724, 0.0499886, 0.0499886, 1 )
		self._add_element("Oxygen", "O", 8, 1.662959, 1.247219, 3.212534, 0.642507, 2, 15.994915, 0.126424, 1, 0.0499886, 0.0499886 )
		self._add_element("Fluorine", "F", 9, 1.587370, 1.154623, 3.269226, 0.653845, 1, 18.998403, 0.146270, 0.699992, 1, 1 )
		self._add_element("Neon", "Ne", 10, 2.494438, 1.322808, 2.910178, 0.582036, 0, 19.992440, 0.000000, 0.699992, 0.889998, 0.959991 )
		self._add_element("Sodium", "Na", 11, 3.968424, 5.782561, 4.289678, 0.428968, 1, 22.989770, 0.034179, 0.669993, 0.359991, 0.949996 )
		self._add_element("Magnesium", "Mg", 12, 2.456644, 2.570027, 3.269226, 0.653845, 2, 23.985042, 0.048144, 0.539986, 1, 0 )
		self._add_element("Aluminium", "Al", 13, 2.929075, 2.229877, 3.873938, 0.774788, 6, 26.981538, 0.059169, 0.749996, 0.649989, 0.649989 )
		self._add_element("Silicon", "Si", 14, 2.645616, 1.770673, 3.968424, 0.793685, 6, 27.976927, 0.069827, 0.499992, 0.6, 0.6 )
		self._add_element("Phosphorus", "P", 15, 2.456644, 1.681856, 3.930630, 0.786126, 5, 30.973762, 0.080485, 1, 0.499992, 0 )
		self._add_element("Sulphur", "S", 16, 2.305466, 1.965315, 3.779452, 0.755890, 4, 31.972071, 0.094818, 1, 1, 0.18999 )
		self._add_element("Chlorine", "Cl", 17, 2.248774, 1.884057, 3.722760, 0.744552, 1, 34.968853, 0.116134, 0.119997, 0.939986, 0.119997 )
		self._add_element("Argon", "Ar", 18, 3.344815, 3.288123, 3.552685, 0.710537, 0, 39.962383, 0.000000, 0.499992, 0.819989, 0.889998 )
		self._add_element("Potassium", "K", 19, 2.891281, 3.836144, 5.196746, 1.039349, 1, 38.963707, 0.030136, 0.559991, 0.249989, 0.829999 )
		self._add_element("Calcium", "Ca", 20, 2.248774, 3.288123, 3.728429, 0.745686, 2, 39.962591, 0.036751, 0.239994, 1, 0 )
		self._add_element("Scandium", "Sc", 21, 3.099151, 2.721205, 3.212534, 0.642507, 6, 44.955910, 0.049982, 0.899992, 0.899992, 0.899992 )
		self._add_element("Titanium", "Ti", 22, 3.155842, 2.494438, 3.212534, 0.642507, 6, 47.947947, 0.056597, 0.749996, 0.759991, 0.779995 )
		self._add_element("Vanadium", "V", 23, 2.891281, 2.305466, 3.212534, 0.642507, 6, 50.943964, 0.059904, 0.649989, 0.649989, 0.669993 )
		self._add_element("Chromium", "Cr", 24, 2.929075, 2.229877, 3.212534, 0.642507, 6, 51.940512, 0.061007, 0.539986, 0.6, 0.779995 )
		self._add_element("Manganese", "Mn", 25, 2.929075, 2.210979, 3.212534, 0.642507, 8, 54.938050, 0.056964, 0.609995, 0.479988, 0.779995 )
		self._add_element("Iron", "Fe", 26, 2.910178, 2.210979, 3.212534, 0.642507, 6, 55.934942, 0.067255, 0.499992, 0.479988, 0.779995 )
		self._add_element("Cobalt", "Co", 27, 2.891281, 2.192082, 3.212534, 0.642507, 6, 58.933200, 0.069092, 0.439994, 0.479988, 0.779995 )
		self._add_element("Nickel", "Ni", 28, 3.212534, 2.173185, 3.080253, 0.616051, 6, 57.935348, 0.070195, 0.359991, 0.479988, 0.759991 )
		self._add_element("Copper", "Cu", 29, 3.250329, 2.210979, 2.645616, 0.529123, 6, 62.929601, 0.069827, 1, 0.479988, 0.379995 )
		self._add_element("Zinc", "Zn", 30, 3.118048, 2.362157, 2.626719, 0.525344, 6, 63.929147, 0.060639, 0.489998, 0.499992, 0.689998 )
		self._add_element("Gallium", "Ga", 31, 2.683411, 2.381055, 3.533788, 0.706758, 3, 68.925581, 0.066520, 0.759991, 0.559991, 0.559991 )
		self._add_element("Germanium", "Ge", 32, 2.588925, 2.244994, 3.212534, 0.642507, 4, 73.921178, 0.073870, 0.4, 0.559991, 0.559991 )
		self._add_element("Arsenic", "As", 33, 2.664514, 2.267671, 3.495993, 0.699199, 3, 74.921596, 0.080118, 0.739986, 0.499992, 0.889998 )
		self._add_element("Selenium", "Se", 34, 2.683411, 2.210979, 3.590479, 0.718096, 2, 79.916522, 0.093716, 1, 0.629999, 0 )
		self._add_element("Bromine", "Br", 35, 2.664514, 2.205310, 3.968424, 0.793685, 1, 78.918338, 0.108784, 0.649989, 0.159991, 0.159991 )
		self._add_element("Krypton", "Kr", 36, 3.987322, 3.609377, 3.817246, 0.763449, 0, 83.911507, 0.000000, 0.359991, 0.719997, 0.819989 )
		self._add_element("Rubidium", "Rb", 37, 3.155842, 4.081808, 3.212534, 0.642507, 1, 84.911789, 0.030136, 0.439994, 0.179995, 0.689998 )
		self._add_element("Strontium", "Sr", 38, 2.494438, 3.609377, 3.212534, 0.642507, 2, 87.905614, 0.034914, 0, 1, 0 )
		self._add_element("Yttrium", "Y", 39, 3.741657, 3.061356, 3.212534, 0.642507, 6, 88.905848, 0.044836, 0.579995, 1, 1 )
		self._add_element("Zirconium", "Zr", 40, 3.325918, 2.740103, 3.212534, 0.642507, 6, 89.904704, 0.048879, 0.579995, 0.879988, 0.879988 )
		self._add_element("Niobium", "Nb", 41, 3.174740, 2.532233, 3.212534, 0.642507, 6, 92.906378, 0.058802, 0.449989, 0.759991, 0.78999 )
		self._add_element("Molybdenum", "Mo", 42, 3.155842, 2.456644, 3.212534, 0.642507, 6, 97.905408, 0.079383, 0.329992, 0.709987, 0.709987 )
		self._add_element("Technetium", "Tc", 43, 2.929075, 2.399952, 3.212534, 0.642507, 6, 96.906365, 0.069827, 0.229999, 0.619989, 0.619989 )
		self._add_element("Ruthenium", "Ru", 44, 3.023562, 2.362157, 3.212534, 0.642507, 6, 101.904349, 0.080853, 0.139986, 0.559991, 0.559991 )
		self._add_element("Rhodium", "Rh", 45, 3.118048, 2.362157, 3.212534, 0.642507, 6, 102.905504, 0.083793, 0.0399939, 0.489998, 0.549996 )
		self._add_element("Palladium", "Pd", 46, 3.212534, 2.418849, 3.080253, 0.616051, 6, 105.903483, 0.080853, 0, 0.409995, 0.519997 )
		self._add_element("Silver", "Ag", 47, 3.382609, 2.532233, 3.250329, 0.650066, 6, 106.905093, 0.070930, 0.879988, 0.879988, 1 )
		self._add_element("Cadmium", "Cd", 48, 3.571582, 2.796794, 2.985767, 0.597153, 6, 113.903358, 0.062110, 1, 0.849989, 0.559991 )
		self._add_element("Indium", "In", 49, 3.458198, 2.721205, 3.647171, 0.729434, 3, 114.903878, 0.065417, 0.649989, 0.459998, 0.449989 )
		self._add_element("Tin", "Sn", 50, 3.136945, 2.617270, 4.100705, 0.820141, 4, 119.902197, 0.072032, 0.4, 0.499992, 0.499992 )
		self._add_element("Antimony", "Sb", 51, 3.136945, 2.645616, 4.157397, 0.831479, 3, 120.903818, 0.075340, 0.619989, 0.38999, 0.709987 )
		self._add_element("Tellurium", "Te", 52, 3.155842, 2.604042, 3.892835, 0.778567, 2, 129.906223, 0.077178, 0.829999, 0.479988, 0 )
		self._add_element("Iodine", "I", 53, 3.023562, 2.621050, 4.062911, 0.812582, 1, 126.904468, 0.097758, 0.579995, 0, 0.579995 )
		self._add_element("Xenon", "Xe", 54, 4.119603, 3.741657, 4.081808, 0.816362, 0, 131.904155, 0.095553, 0.259998, 0.619989, 0.689998 )
		self._add_element("Cesium", "Cs", 55, 3.533788, 4.440856, 3.212534, 0.642507, 1, 132.905447, 0.029033, 0.339986, 0.0899977, 0.559991 )
		self._add_element("Barium", "Ba", 56, 2.910178, 3.741657, 3.212534, 0.642507, 2, 137.905241, 0.032709, 0, 0.78999, 0 )
		self._add_element("Lanthanum", "La", 57, 3.911733, 3.193637, 3.212534, 0.642507, 12, 138.906348, 0.040426, 0.439994, 0.829999, 1 )
		self._add_element("Cerium", "Ce", 58, 3.836144, 3.458198, 3.212534, 0.642507, 6, 139.905434, 0.041161, 1, 1, 0.779995 )
		self._add_element("Praseodymium", "Pr", 59, 3.817246, 3.439301, 3.212534, 0.642507, 6, 140.907648, 0.041529, 0.849989, 1, 0.779995 )
		self._add_element("Neodymium", "Nd", 60, 3.798349, 3.420404, 3.212534, 0.642507, 6, 141.907719, 0.041896, 0.779995, 1, 0.779995 )
		self._add_element("Promethium", "Pm", 61, 3.779452, 3.401507, 3.212534, 0.642507, 6, 144.912744, 0.041529, 0.639994, 1, 0.779995 )
		self._add_element("Samarium", "Sm", 62, 3.779452, 3.401507, 3.212534, 0.642507, 6, 151.919728, 0.042999, 0.559991, 1, 0.779995 )
		self._add_element("Europium", "Eu", 63, 4.138500, 3.760555, 3.212534, 0.642507, 6, 152.921226, 0.044101, 0.379995, 1, 0.779995 )
		self._add_element("Gadolinium", "Gd", 64, 3.760555, 3.382609, 3.212534, 0.642507, 6, 157.924101, 0.044101, 0.269993, 1, 0.779995 )
		self._add_element("Terbium", "Tb", 65, 3.703863, 3.325918, 3.212534, 0.642507, 6, 158.925343, 0.040426, 0.18999, 1, 0.779995 )
		self._add_element("Dysprosium", "Dy", 66, 3.684966, 3.307020, 3.212534, 0.642507, 6, 163.929171, 0.044836, 0.119997, 1, 0.779995 )
		self._add_element("Holmium", "Ho", 67, 3.666068, 3.288123, 3.212534, 0.642507, 6, 164.930319, 0.045204, 0, 1, 0.609995 )
		self._add_element("Erbium", "Er", 68, 3.647171, 3.269226, 3.212534, 0.642507, 6, 165.930290, 0.045571, 0, 0.899992, 0.459998 )
		self._add_element("Thulium", "Tm", 69, 3.628274, 3.250329, 3.212534, 0.642507, 6, 168.934211, 0.045939, 0, 0.829999, 0.319997 )
		self._add_element("Ytterbium", "Yb", 70, 4.044014, 3.666068, 3.212534, 0.642507, 6, 173.938858, 0.040426, 0, 0.749996, 0.219989 )
		self._add_element("Lutetium", "Lu", 71, 3.628274, 3.250329, 3.212534, 0.642507, 6, 174.940768, 0.046674, 0, 0.669993, 0.139986 )
		self._add_element("Hafnium", "Hf", 72, 3.344815, 2.721205, 3.212534, 0.642507, 6, 179.946549, 0.047777, 0.299992, 0.759991, 1 )
		self._add_element("Tantalum", "Ta", 73, 3.080253, 2.532233, 3.212534, 0.642507, 6, 180.947996, 0.055127, 0.299992, 0.649989, 1 )
		self._add_element("Tungsten", "W", 74, 2.966870, 2.456644, 3.212534, 0.642507, 6, 183.950933, 0.086733, 0.129992, 0.579995, 0.839994 )
		self._add_element("Rhenium", "Re", 75, 2.929075, 2.418849, 3.212534, 0.642507, 6, 186.955751, 0.069827, 0.149996, 0.489998, 0.669993 )
		self._add_element("Osmium", "Os", 76, 2.966870, 2.381055, 3.212534, 0.642507, 6, 191.961479, 0.080853, 0.149996, 0.4, 0.58999 )
		self._add_element("Iridium", "Ir", 77, 2.872383, 2.399952, 3.212534, 0.642507, 6, 192.962924, 0.080853, 0.0899977, 0.329992, 0.529992 )
		self._add_element("Platinum", "Pt", 78, 3.212534, 2.456644, 3.250329, 0.650066, 6, 194.964774, 0.083793, 0.959991, 0.929992, 0.819989 )
		self._add_element("Gold", "Au", 79, 3.212534, 2.532233, 3.136945, 0.627389, 6, 196.966552, 0.093348, 0.8, 0.819989, 0.119997 )
		self._add_element("Mercury", "Hg", 80, 3.590479, 2.815692, 2.929075, 0.585815, 6, 201.970626, 0.073502, 0.709987, 0.709987, 0.759991 )
		self._add_element("Thallium", "Tl", 81, 3.307020, 2.796794, 3.703863, 0.740773, 3, 204.974412, 0.074972, 0.649989, 0.329992, 0.299992 )
		self._add_element("Lead", "Pb", 82, 3.288123, 2.796794, 3.817246, 0.763449, 4, 207.976636, 0.085630, 0.339986, 0.349996, 0.379995 )
		self._add_element("Bismuth", "Bi", 83, 3.288123, 2.740103, 3.212534, 0.642507, 3, 208.980383, 0.074237, 0.619989, 0.309987, 0.709987 )
		self._add_element("Polonium", "Po", 84, 3.552685, 2.759000, 3.212534, 0.642507, 2, 208.982416, 0.073502, 0.669993, 0.359991, 0 )
		self._add_element("Astatine", "At", 85, 3.590479, 2.740103, 3.212534, 0.642507, 1, 209.987131, 0.080853, 0.459998, 0.309987, 0.269993 )
		self._add_element("Radon", "Rn", 86, 4.913287, 4.535342, 3.212534, 0.642507, 0, 210.990585, 0.000000, 0.259998, 0.509987, 0.58999 )
		self._add_element("Francium", "Fr", 87, 4.157397, 3.779452, 3.212534, 0.642507, 1, 223.019731, 0.025726, 0.259998, 0, 0.4 )
		self._add_element("radium", "Ra", 88, 3.968424, 3.590479, 3.212534, 0.642507, 2, 223.018497, 0.032709, 0, 0.489998, 0 )
		self._add_element("Actinium", "Ac", 89, 3.930630, 3.552685, 3.212534, 0.642507, 6, 227.027747, 0.040426, 0.439994, 0.669993, 0.979995 )
		self._add_element("Thorium", "Th", 90, 3.760555, 3.382609, 3.212534, 0.642507, 6, 232.038050, 0.047777, 0, 0.729992, 1 )
		self._add_element("Protactinium", "Pa", 91, 3.420404, 3.042459, 3.212534, 0.642507, 6, 231.035879, 0.055127, 0, 0.629999, 1 )
		self._add_element("Uranium", "U", 92, 3.363712, 2.985767, 3.514890, 0.702978, 6, 238.050783, 0.050717, 0, 0.559991, 1 )
		self._add_element("Neptunium", "Np", 93, 3.307020, 2.929075, 3.212534, 0.642507, 6, 237.048167, 0.049982, 0, 0.499992, 1 )
		self._add_element("Plutionium", "Pu", 94, 3.269226, 2.891281, 3.212534, 0.642507, 6, 238.049553, 0.047042, 0, 0.419989, 1 )
		self._add_element("Americium", "Am", 95, 3.231431, 2.022007, 3.212534, 0.642507, 6, 241.056823, 0.047777, 0.329992, 0.359991, 0.949996 )
		self._add_element("Curium", "Cm", 96, 3.212534, 0.000000, 3.212534, 0.642507, 6, 243.061382, 0.047777, 0.469993, 0.359991, 0.889998 )
		self._add_element("Berkelium", "Bk", 97, 3.212534, 0.000000, 3.212534, 0.642507, 6, 247.070299, 0.047777, 0.539986, 0.309987, 0.889998 )
		self._add_element("Californium", "Cf", 98, 3.212534, 0.000000, 3.212534, 0.642507, 6, 249.074847, 0.047777, 0.629999, 0.209995, 0.829999 )
		self._add_element("Einsteinium", "Es", 99, 3.212534, 0.000000, 3.212534, 0.642507, 6, 252.082970, 0.047777, 0.699992, 0.119997, 0.829999 )
		self._add_element("Fermium", "Fm", 100, 3.212534, 0.000000, 3.212534, 0.642507, 6, 257.095099, 0.047777, 0.699992, 0.119997, 0.729992 )
		self._add_element("Mendelevium", "Md", 101, 3.212534, 0.000000, 3.212534, 0.642507, 6, 256.094050, 0.047777, 0.699992, 0.0499886, 0.649989 )
		self._add_element("Nobelium", "No", 102, 3.212534, 0.000000, 3.212534, 0.642507, 6, 259.101020, 0.047777, 0.739986, 0.0499886, 0.529992 )
		self._add_element("Lawrencium", "Lr", 103, 3.212534, 0.000000, 3.212534, 0.642507, 6, 262.109690, 0.047777, 0.779995, 0, 0.4 )
		self._add_element("Rutherfordium", "Rf", 104, 3.401507, 0.000000, 3.212534, 0.642507, 6, 261.108750, 0.000000, 0.8, 0, 0.349996 )
		self._add_element("Dubnium", "Db", 105, 3.401507, 0.000000, 3.212534, 0.642507, 6, 262.114150, 0.000000, 0.819989, 0, 0.309987 )
		self._add_element("Seaborgium", "Sg", 106, 3.401507, 0.000000, 3.212534, 0.642507, 6, 266.121930, 0.000000, 0.849989, 0, 0.269993 )
		self._add_element("Bohrium", "Bh", 107, 3.401507, 0.000000, 3.212534, 0.642507, 6, 264.124730, 0.000000, 0.879988, 0, 0.219989 )
		self._add_element("Hassium", "Hs", 108, 3.401507, 0.000000, 3.212534, 0.642507, 6, 277.000000, 0.000000, 0.899992, 0, 0.179995 )
		self._add_element("Meitnerium", "Mt", 109, 3.401507, 0.000000, 3.212534, 0.642507, 6, 268.138820, 0.000000, 0.919997, 0, 0.149996 )
		self._add_element("Dummy", "Xx", 0, 0.377945, 0.000000, 0.000000, 0.000000, 0, 0.000000, 0.000000, 0.0699931, 0.499992, 0.699992 )
		self._add_element("Dummy", "X", 0, 0.377945, 0.000000, 0.000000, 0.000000, 0, 0.000000, 0.000000, 0.0699931, 0.499992, 0.699992 )
		self._add_element("Tv", "Tv", 112, 0.377945, 0.000000, 0.000000, 0.000000, 0, 1.000000, 0.000000, 0.0699931, 0.499992, 0.699992 )

	def _add_all_isotopes(self):
		self._add_isotope("H",1, 1.007825, 99.988500 )
		self._add_isotope("H",2, 2.014102, 0.011500 )
		self._add_isotope("H",3, 3.016049, 0.000000 )
		self._add_isotope("He",4, 4.002603, 99.999863 )
		self._add_isotope("He",3, 3.016029, 0.000137 )
		self._add_isotope("Li",7, 7.016004, 92.410000 )
		self._add_isotope("Li",6, 6.015122, 7.590000 )
		self._add_isotope("Be",9, 9.012182, 100.000000 )
		self._add_isotope("B",11, 11.009305, 80.100000 )
		self._add_isotope("B",10, 10.012937, 19.900000 )
		self._add_isotope("C",12, 12.000000, 98.930000 )
		self._add_isotope("C",13, 13.003355, 1.070000 )
		self._add_isotope("C",14, 14.003242, 0.000000 )
		self._add_isotope("N",14, 14.003074, 99.632000 )
		self._add_isotope("N",15, 15.000109, 0.368000 )
		self._add_isotope("O",16, 15.994915, 99.757000 )
		self._add_isotope("O",18, 17.999160, 0.205000 )
		self._add_isotope("O",17, 16.999132, 0.038000 )
		self._add_isotope("F",19, 18.998403, 100.000000 )
		self._add_isotope("Ne",20, 19.992440, 90.480000 )
		self._add_isotope("Ne",22, 21.991386, 9.250000 )
		self._add_isotope("Ne",21, 20.993847, 0.270000 )
		self._add_isotope("Na",23, 22.989770, 100.000000 )
		self._add_isotope("Mg",24, 23.985042, 78.990000 )
		self._add_isotope("Mg",26, 25.982593, 11.010000 )
		self._add_isotope("Mg",25, 24.985837, 10.000000 )
		self._add_isotope("Al",27, 26.981538, 100.000000 )
		self._add_isotope("Si",28, 27.976927, 92.229700 )
		self._add_isotope("Si",29, 28.976495, 4.683200 )
		self._add_isotope("Si",30, 29.973770, 3.087200 )
		self._add_isotope("P",31, 30.973762, 100.000000 )
		self._add_isotope("S",32, 31.972071, 94.930000 )
		self._add_isotope("S",34, 33.967867, 4.290000 )
		self._add_isotope("S",33, 32.971458, 0.760000 )
		self._add_isotope("S",36, 35.967081, 0.020000 )
		self._add_isotope("Cl",35, 34.968853, 75.780000 )
		self._add_isotope("Cl",37, 36.965903, 24.220000 )
		self._add_isotope("Ar",40, 39.962383, 99.600300 )
		self._add_isotope("Ar",36, 35.967546, 0.336500 )
		self._add_isotope("Ar",38, 37.962732, 0.063200 )
		self._add_isotope("K",39, 38.963707, 93.258100 )
		self._add_isotope("K",41, 40.961826, 6.730200 )
		self._add_isotope("K",40, 39.963999, 0.011700 )
		self._add_isotope("Ca",40, 39.962591, 96.941000 )
		self._add_isotope("Ca",44, 43.955481, 2.086000 )
		self._add_isotope("Ca",42, 41.958618, 0.647000 )
		self._add_isotope("Ca",48, 47.952534, 0.187000 )
		self._add_isotope("Ca",43, 42.958767, 0.135000 )
		self._add_isotope("Ca",46, 45.953693, 0.004000 )
		self._add_isotope("Sc",45, 44.955910, 100.000000 )
		self._add_isotope("Ti",48, 47.947947, 73.720000 )
		self._add_isotope("Ti",46, 45.952630, 8.250000 )
		self._add_isotope("Ti",47, 46.951764, 7.440000 )
		self._add_isotope("Ti",49, 48.947871, 5.410000 )
		self._add_isotope("Ti",50, 49.944792, 5.180000 )
		self._add_isotope("V",51, 50.943964, 99.750000 )
		self._add_isotope("V",50, 49.947163, 0.250000 )
		self._add_isotope("Cr",52, 51.940512, 83.789000 )
		self._add_isotope("Cr",53, 52.940654, 9.501000 )
		self._add_isotope("Cr",50, 49.946050, 4.345000 )
		self._add_isotope("Cr",54, 53.938885, 2.365000 )
		self._add_isotope("Mn",55, 54.938050, 100.000000 )
		self._add_isotope("Fe",56, 55.934942, 91.754000 )
		self._add_isotope("Fe",54, 53.939615, 5.845000 )
		self._add_isotope("Fe",57, 56.935399, 2.119000 )
		self._add_isotope("Fe",58, 57.933281, 0.282000 )
		self._add_isotope("Co",59, 58.933200, 100.000000 )
		self._add_isotope("Ni",58, 57.935348, 68.076900 )
		self._add_isotope("Ni",60, 59.930791, 26.223100 )
		self._add_isotope("Ni",62, 61.928349, 3.634500 )
		self._add_isotope("Ni",61, 60.931060, 1.139900 )
		self._add_isotope("Ni",64, 63.927970, 0.925600 )
		self._add_isotope("Cu",63, 62.929601, 69.170000 )
		self._add_isotope("Cu",65, 64.927794, 30.830000 )
		self._add_isotope("Zn",64, 63.929147, 48.630000 )
		self._add_isotope("Zn",66, 65.926037, 27.900000 )
		self._add_isotope("Zn",68, 67.924848, 18.750000 )
		self._add_isotope("Zn",67, 66.927131, 4.100000 )
		self._add_isotope("Zn",70, 69.925325, 0.620000 )
		self._add_isotope("Ga",69, 68.925581, 60.108000 )
		self._add_isotope("Ga",71, 70.924705, 39.892000 )
		self._add_isotope("Ge",74, 73.921178, 36.280000 )
		self._add_isotope("Ge",72, 71.922076, 27.540000 )
		self._add_isotope("Ge",70, 69.924250, 20.840000 )
		self._add_isotope("Ge",73, 72.923459, 7.730000 )
		self._add_isotope("Ge",76, 75.921403, 7.610000 )
		self._add_isotope("As",75, 74.921596, 100.000000 )
		self._add_isotope("Se",80, 79.916522, 49.610000 )
		self._add_isotope("Se",78, 77.917310, 23.770000 )
		self._add_isotope("Se",76, 75.919214, 9.370000 )
		self._add_isotope("Se",82, 81.916700, 8.730000 )
		self._add_isotope("Se",77, 76.919915, 7.630000 )
		self._add_isotope("Se",74, 73.922477, 0.890000 )
		self._add_isotope("Br",79, 78.918338, 50.690000 )
		self._add_isotope("Br",81, 80.916291, 49.310000 )
		self._add_isotope("Kr",84, 83.911507, 57.000000 )
		self._add_isotope("Kr",86, 85.910610, 17.300000 )
		self._add_isotope("Kr",82, 81.913485, 11.580000 )
		self._add_isotope("Kr",83, 82.914136, 11.490000 )
		self._add_isotope("Kr",80, 79.916378, 2.280000 )
		self._add_isotope("Kr",78, 77.920386, 0.350000 )
		self._add_isotope("Rb",85, 84.911789, 72.170000 )
		self._add_isotope("Rb",87, 86.909183, 27.830000 )
		self._add_isotope("Sr",88, 87.905614, 82.580000 )
		self._add_isotope("Sr",86, 85.909262, 9.860000 )
		self._add_isotope("Sr",87, 86.908879, 7.000000 )
		self._add_isotope("Sr",84, 83.913425, 0.560000 )
		self._add_isotope("Y",89, 88.905848, 100.000000 )
		self._add_isotope("Zr",90, 89.904704, 51.450000 )
		self._add_isotope("Zr",94, 93.906316, 17.380000 )
		self._add_isotope("Zr",92, 91.905040, 17.150000 )
		self._add_isotope("Zr",91, 90.905645, 11.220000 )
		self._add_isotope("Zr",96, 95.908276, 2.800000 )
		self._add_isotope("Nb",93, 92.906378, 100.000000 )
		self._add_isotope("Mo",98, 97.905408, 24.130000 )
		self._add_isotope("Mo",96, 95.904679, 16.680000 )
		self._add_isotope("Mo",95, 94.905841, 15.920000 )
		self._add_isotope("Mo",92, 91.906810, 14.840000 )
		self._add_isotope("Mo",100, 99.907477, 9.630000 )
		self._add_isotope("Mo",97, 96.906021, 9.550000 )
		self._add_isotope("Mo",94, 93.905088, 9.250000 )
		self._add_isotope("Tc",97, 96.906365, 33.333333 )
		self._add_isotope("Tc",98, 97.907216, 33.333333 )
		self._add_isotope("Tc",99, 98.906255, 33.333333 )
		self._add_isotope("Ru",102, 101.904349, 31.550000 )
		self._add_isotope("Ru",104, 103.905430, 18.620000 )
		self._add_isotope("Ru",101, 100.905582, 17.060000 )
		self._add_isotope("Ru",99, 98.905939, 12.760000 )
		self._add_isotope("Ru",100, 99.904220, 12.600000 )
		self._add_isotope("Ru",96, 95.907598, 5.540000 )
		self._add_isotope("Ru",98, 97.905287, 1.870000 )
		self._add_isotope("Rh",103, 102.905504, 100.000000 )
		self._add_isotope("Pd",106, 105.903483, 27.330000 )
		self._add_isotope("Pd",108, 107.903894, 26.460000 )
		self._add_isotope("Pd",105, 104.905084, 22.330000 )
		self._add_isotope("Pd",110, 109.905152, 11.720000 )
		self._add_isotope("Pd",104, 103.904035, 11.140000 )
		self._add_isotope("Pd",102, 101.905608, 1.020000 )
		self._add_isotope("Ag",107, 106.905093, 51.839000 )
		self._add_isotope("Ag",109, 108.904756, 48.161000 )
		self._add_isotope("Cd",114, 113.903358, 28.730000 )
		self._add_isotope("Cd",112, 111.902757, 24.130000 )
		self._add_isotope("Cd",111, 110.904182, 12.800000 )
		self._add_isotope("Cd",110, 109.903006, 12.490000 )
		self._add_isotope("Cd",113, 112.904401, 12.220000 )
		self._add_isotope("Cd",116, 115.904755, 7.490000 )
		self._add_isotope("Cd",106, 105.906458, 1.250000 )
		self._add_isotope("Cd",108, 107.904183, 0.890000 )
		self._add_isotope("In",115, 114.903878, 95.710000 )
		self._add_isotope("In",113, 112.904061, 4.290000 )
		self._add_isotope("Sn",120, 119.902197, 32.580000 )
		self._add_isotope("Sn",118, 117.901606, 24.220000 )
		self._add_isotope("Sn",116, 115.901744, 14.540000 )
		self._add_isotope("Sn",119, 118.903309, 8.590000 )
		self._add_isotope("Sn",117, 116.902954, 7.680000 )
		self._add_isotope("Sn",124, 123.905275, 5.790000 )
		self._add_isotope("Sn",122, 121.903440, 4.630000 )
		self._add_isotope("Sn",112, 111.904821, 0.970000 )
		self._add_isotope("Sn",114, 113.902782, 0.660000 )
		self._add_isotope("Sn",115, 114.903346, 0.340000 )
		self._add_isotope("Sb",121, 120.903818, 57.210000 )
		self._add_isotope("Sb",123, 122.904216, 42.790000 )
		self._add_isotope("Te",130, 129.906223, 34.080000 )
		self._add_isotope("Te",128, 127.904461, 31.740000 )
		self._add_isotope("Te",126, 125.903306, 18.840000 )
		self._add_isotope("Te",125, 124.904425, 7.070000 )
		self._add_isotope("Te",124, 123.902820, 4.740000 )
		self._add_isotope("Te",122, 121.903047, 2.550000 )
		self._add_isotope("Te",123, 122.904273, 0.890000 )
		self._add_isotope("Te",120, 119.904020, 0.090000 )
		self._add_isotope("I",127, 126.904468, 100.000000 )
		self._add_isotope("Xe",132, 131.904155, 26.890000 )
		self._add_isotope("Xe",129, 128.904779, 26.440000 )
		self._add_isotope("Xe",131, 130.905082, 21.180000 )
		self._add_isotope("Xe",134, 133.905394, 10.440000 )
		self._add_isotope("Xe",136, 135.907220, 8.870000 )
		self._add_isotope("Xe",130, 129.903508, 4.080000 )
		self._add_isotope("Xe",128, 127.903530, 1.920000 )
		self._add_isotope("Xe",126, 125.904269, 0.090000 )
		self._add_isotope("Xe",124, 123.905896, 0.090000 )
		self._add_isotope("Cs",133, 132.905447, 100.000000 )
		self._add_isotope("Ba",138, 137.905241, 71.698000 )
		self._add_isotope("Ba",137, 136.905821, 11.232000 )
		self._add_isotope("Ba",136, 135.904570, 7.854000 )
		self._add_isotope("Ba",135, 134.905683, 6.592000 )
		self._add_isotope("Ba",134, 133.904503, 2.417000 )
		self._add_isotope("Ba",130, 129.906310, 0.106000 )
		self._add_isotope("Ba",132, 131.905056, 0.101000 )
		self._add_isotope("La",139, 138.906348, 99.910000 )
		self._add_isotope("La",138, 137.907107, 0.090000 )
		self._add_isotope("Ce",140, 139.905434, 88.450000 )
		self._add_isotope("Ce",142, 141.909240, 11.114000 )
		self._add_isotope("Ce",138, 137.905986, 0.251000 )
		self._add_isotope("Ce",136, 135.907140, 0.185000 )
		self._add_isotope("Pr",141, 140.907648, 100.000000 )
		self._add_isotope("Nd",142, 141.907719, 27.200000 )
		self._add_isotope("Nd",144, 143.910083, 23.800000 )
		self._add_isotope("Nd",146, 145.913112, 17.200000 )
		self._add_isotope("Nd",143, 142.909810, 12.200000 )
		self._add_isotope("Nd",145, 144.912569, 8.300000 )
		self._add_isotope("Nd",148, 147.916889, 5.700000 )
		self._add_isotope("Nd",150, 149.920887, 5.600000 )
		self._add_isotope("Pm",145, 144.912744, 50.000000 )
		self._add_isotope("Pm",147, 146.915134, 50.000000 )
		self._add_isotope("Sm",152, 151.919728, 26.750000 )
		self._add_isotope("Sm",154, 153.922205, 22.750000 )
		self._add_isotope("Sm",147, 146.914893, 14.990000 )
		self._add_isotope("Sm",149, 148.917180, 13.820000 )
		self._add_isotope("Sm",148, 147.914818, 11.240000 )
		self._add_isotope("Sm",150, 149.917271, 7.380000 )
		self._add_isotope("Sm",144, 143.911995, 3.070000 )
		self._add_isotope("Eu",153, 152.921226, 52.190000 )
		self._add_isotope("Eu",151, 150.919846, 47.810000 )
		self._add_isotope("Gd",158, 157.924101, 24.840000 )
		self._add_isotope("Gd",160, 159.927051, 21.860000 )
		self._add_isotope("Gd",156, 155.922120, 20.470000 )
		self._add_isotope("Gd",157, 156.923957, 15.650000 )
		self._add_isotope("Gd",155, 154.922619, 14.800000 )
		self._add_isotope("Gd",154, 153.920862, 2.180000 )
		self._add_isotope("Gd",152, 151.919788, 0.200000 )
		self._add_isotope("Tb",159, 158.925343, 100.000000 )
		self._add_isotope("Dy",164, 163.929171, 28.180000 )
		self._add_isotope("Dy",162, 161.926795, 25.510000 )
		self._add_isotope("Dy",163, 162.928728, 24.900000 )
		self._add_isotope("Dy",161, 160.926930, 18.910000 )
		self._add_isotope("Dy",160, 159.925194, 2.340000 )
		self._add_isotope("Dy",158, 157.924405, 0.100000 )
		self._add_isotope("Dy",156, 155.924278, 0.060000 )
		self._add_isotope("Ho",165, 164.930319, 100.000000 )
		self._add_isotope("Er",166, 165.930290, 33.610000 )
		self._add_isotope("Er",168, 167.932368, 26.780000 )
		self._add_isotope("Er",167, 166.932045, 22.930000 )
		self._add_isotope("Er",170, 169.935460, 14.930000 )
		self._add_isotope("Er",164, 163.929197, 1.610000 )
		self._add_isotope("Er",162, 161.928775, 0.140000 )
		self._add_isotope("Tm",169, 168.934211, 100.000000 )
		self._add_isotope("Yb",174, 173.938858, 31.830000 )
		self._add_isotope("Yb",172, 171.936378, 21.830000 )
		self._add_isotope("Yb",173, 172.938207, 16.130000 )
		self._add_isotope("Yb",171, 170.936322, 14.280000 )
		self._add_isotope("Yb",176, 175.942568, 12.760000 )
		self._add_isotope("Yb",170, 169.934759, 3.040000 )
		self._add_isotope("Yb",168, 167.933894, 0.130000 )
		self._add_isotope("Lu",175, 174.940768, 97.410000 )
		self._add_isotope("Lu",176, 175.942682, 2.590000 )
		self._add_isotope("Hf",180, 179.946549, 35.080000 )
		self._add_isotope("Hf",178, 177.943698, 27.280000 )
		self._add_isotope("Hf",177, 176.943220, 18.600000 )
		self._add_isotope("Hf",179, 178.945815, 13.620000 )
		self._add_isotope("Hf",176, 175.941402, 5.260000 )
		self._add_isotope("Hf",174, 173.940040, 0.160000 )
		self._add_isotope("Ta",181, 180.947996, 99.988000 )
		self._add_isotope("Ta",180, 179.947466, 0.012000 )
		self._add_isotope("W",184, 183.950933, 30.640000 )
		self._add_isotope("W",186, 185.954362, 28.430000 )
		self._add_isotope("W",182, 181.948206, 26.500000 )
		self._add_isotope("W",183, 182.950224, 14.310000 )
		self._add_isotope("W",180, 179.946706, 0.120000 )
		self._add_isotope("Re",187, 186.955751, 62.600000 )
		self._add_isotope("Re",185, 184.952956, 37.400000 )
		self._add_isotope("Os",192, 191.961479, 40.780000 )
		self._add_isotope("Os",190, 189.958445, 26.260000 )
		self._add_isotope("Os",189, 188.958145, 16.150000 )
		self._add_isotope("Os",188, 187.955836, 13.240000 )
		self._add_isotope("Os",187, 186.955748, 1.960000 )
		self._add_isotope("Os",186, 185.953838, 1.590000 )
		self._add_isotope("Os",184, 183.952491, 0.020000 )
		self._add_isotope("Ir",193, 192.962924, 62.700000 )
		self._add_isotope("Ir",191, 190.960591, 37.300000 )
		self._add_isotope("Pt",195, 194.964774, 33.832000 )
		self._add_isotope("Pt",194, 193.962664, 32.967000 )
		self._add_isotope("Pt",196, 195.964935, 25.242000 )
		self._add_isotope("Pt",198, 197.967876, 7.163000 )
		self._add_isotope("Pt",192, 191.961035, 0.782000 )
		self._add_isotope("Pt",190, 189.959930, 0.014000 )
		self._add_isotope("Au",197, 196.966552, 100.000000 )
		self._add_isotope("Hg",202, 201.970626, 29.860000 )
		self._add_isotope("Hg",200, 199.968309, 23.100000 )
		self._add_isotope("Hg",199, 198.968262, 16.870000 )
		self._add_isotope("Hg",201, 200.970285, 13.180000 )
		self._add_isotope("Hg",198, 197.966752, 9.970000 )
		self._add_isotope("Hg",204, 203.973476, 6.870000 )
		self._add_isotope("Hg",196, 195.965815, 0.150000 )
		self._add_isotope("Tl",205, 204.974412, 70.476000 )
		self._add_isotope("Tl",203, 202.972329, 29.524000 )
		self._add_isotope("Pb",208, 207.976636, 52.400000 )
		self._add_isotope("Pb",206, 205.974449, 24.100000 )
		self._add_isotope("Pb",207, 206.975881, 22.100000 )
		self._add_isotope("Pb",204, 203.973029, 1.400000 )
		self._add_isotope("Bi",209, 208.980383, 100.000000 )
		self._add_isotope("Po",209, 208.982416, 50.000000 )
		self._add_isotope("Po",210, 209.982857, 50.000000 )
		self._add_isotope("At",210, 209.987131, 50.000000 )
		self._add_isotope("At",211, 210.987481, 50.000000 )
		self._add_isotope("Rn",211, 210.990585, 33.333333 )
		self._add_isotope("Rn",220, 220.011384, 33.333333 )
		self._add_isotope("Rn",222, 222.017571, 33.333333 )
		self._add_isotope("Fr",223, 223.019731, 100.000000 )
		self._add_isotope("Ra",223, 223.018497, 25.000000 )
		self._add_isotope("Ra",224, 224.020202, 25.000000 )
		self._add_isotope("Ra",226, 226.025403, 25.000000 )
		self._add_isotope("Ra",228, 228.031064, 25.000000 )
		self._add_isotope("Ac",227, 227.027747, 100.000000 )
		self._add_isotope("Th",232, 232.038050, 100.000000 )
		self._add_isotope("Th",230, 230.033127, 0.000000 )
		self._add_isotope("Pa",231, 231.035879, 100.000000 )
		self._add_isotope("U",238, 238.050783, 99.274500 )
		self._add_isotope("U",235, 235.043923, 0.720000 )
		self._add_isotope("U",234, 234.040946, 0.005500 )
		self._add_isotope("U",236, 236.045562, 0.000000 )
		self._add_isotope("U",233, 233.039628, 0.000000 )
		self._add_isotope("Np",237, 237.048167, 50.000000 )
		self._add_isotope("Np",239, 239.052931, 50.000000 )
		self._add_isotope("Pu",238, 238.049553, 16.666667 )
		self._add_isotope("Pu",239, 239.052156, 16.666667 )
		self._add_isotope("Pu",240, 240.053808, 16.666667 )
		self._add_isotope("Pu",241, 241.056845, 16.666667 )
		self._add_isotope("Pu",242, 242.058737, 16.666667 )
		self._add_isotope("Pu",244, 244.064198, 16.666667 )
		self._add_isotope("Am",241, 241.056823, 50.000000 )
		self._add_isotope("Am",243, 243.061373, 50.000000 )
		self._add_isotope("Cm",243, 243.061382, 16.666667 )
		self._add_isotope("Cm",244, 244.062746, 16.666667 )
		self._add_isotope("Cm",245, 245.065486, 16.666667 )
		self._add_isotope("Cm",246, 246.067218, 16.666667 )
		self._add_isotope("Cm",247, 247.070347, 16.666667 )
		self._add_isotope("Cm",248, 248.072342, 16.666667 )
		self._add_isotope("Bk",247, 247.070299, 50.000000 )
		self._add_isotope("Bk",249, 249.074980, 50.000000 )
		self._add_isotope("Cf",249, 249.074847, 25.000000 )
		self._add_isotope("Cf",250, 250.076400, 25.000000 )
		self._add_isotope("Cf",251, 251.079580, 25.000000 )
		self._add_isotope("Cf",252, 252.081620, 25.000000 )
		self._add_isotope("Es",252, 252.082970, 100.000000 )
		self._add_isotope("Fm",257, 257.095099, 100.000000 )
		self._add_isotope("Md",256, 256.094050, 50.000000 )
		self._add_isotope("Md",258, 258.098425, 50.000000 )
		self._add_isotope("No",259, 259.101020, 100.000000 )
		self._add_isotope("Lr",262, 262.109690, 100.000000 )
		self._add_isotope("Rf",261, 261.108750, 100.000000 )
		self._add_isotope("Db",262, 262.114150, 100.000000 )
		self._add_isotope("Sg",266, 266.121930, 100.000000 )
		self._add_isotope("Bh",264, 264.124730, 100.000000 )
		self._add_isotope("Hs",277, 277.000000, 100.000000 )
		self._add_isotope("Mt",268, 268.138820, 100.000000 )
