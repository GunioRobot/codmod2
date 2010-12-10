def stdize(x) :
	import numpy
	return (x-numpy.mean(x))/numpy.std(x)

import stata_to_python

def load_maternal() :
	'''
	Output format: region, country, year, age, y, se, x0...xi-1, w0...wi-1
		Where 	y = ln(death rate)
				se = sampling variance in ln(rate) space
				xi = covariates (and datatype dummies)
				wi = covariate standard error
	'''

	# load in the maternal dataset
	import numpy
	# maternal_raw = stata_to_python.genfromdta('/home/j/Project/Causes of Death/CoDMod/pymc-space-time-model/maternal_paper_db_no_missing.dta', missing_flt=numpy.nan)
	maternal_raw = stata_to_python.genfromdta('/home/j/Project/Causes of Death/CoDMod/pymc-space-time-model/sample_dataset.dta', missing_flt=numpy.nan)

	'''
	# create dummy variables by age/region
	age_dummies = numpy.array(zip(maternal_raw['year'], [(maternal_raw['age'] == a).astype(float) for a in numpy.unique(maternal_raw['age'])]))
	
	age_dummies = (maternal_raw['age'][:, None] == numpy.unique(maternal_raw['age'])).astype(float)
	age_dummies.dtype = [('age_' + str(a.__int__()), 'float64') for a in numpy.unique(maternal_raw['age'])]
		
	maternal_raw = numpy.concatenate((maternal_raw, age_dummies), axis=1)

	age_dummies = numpy.array(tuple(age_dummies[x]) for x in range(len(age_dummies)))
	'''
	
	# return the appropriately shaped array
	'''	x0: ln_LDI, x1: ln_LDI_sq, x2: educ, x3: neonatal_mort, x4: tfr, x5: year, x6: age15, x7: age20, x8: age25, x9: age35, x10: age40, x11: age45 '''
	return numpy.array(zip(maternal_raw['region'], maternal_raw['iso3'], stdize(maternal_raw['year']), stdize(maternal_raw['age']), maternal_raw['ln_rate'], maternal_raw['ln_rate_sd'], stdize(numpy.log(maternal_raw['LDI_id'])), stdize(numpy.log(maternal_raw['LDI_id'])**2), stdize(maternal_raw['educ']), stdize(maternal_raw['q_nn_med']), stdize(maternal_raw['tfr']), stdize(maternal_raw['year']), (maternal_raw['age']==15).astype(float), (maternal_raw['age']==20).astype(float), (maternal_raw['age']==25).astype(float), (maternal_raw['age']==35).astype(float), (maternal_raw['age']==40).astype(float), (maternal_raw['age']==45).astype(float),	numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)),	numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw))), dtype=[('region','|S35'),('country','|S3'),('year','<f8'),('age','<f8'),('y','<f8'),('se','<f8'),('x0','<f8'),('x1','<f8'),('x2','<f8'),('x3','<f8'),('x4','<f8'),('x5','<f8'),('x6','<f8'),('x7','<f8'),('x8','<f8'),('x9','<f8'),('x10','<f8'),('x11','<f8'),('w0','<f8'),('w1','<f8'),('w2','<f8'),('w3','<f8'),('w4','<f8'),('w5','<f8'),('w6','<f8'),('w7','<f8'),('w8','<f8'),('w9','<f8'),('w10','<f8'),('w11','<f8')]).view(numpy.recarray)

# run the model	
data = load_maternal()
from model import *
mod_mc = gp_re_a(data)
iter = 20000
mod_mc.sample(iter, iter/2, iter/2000, verbose=1)










