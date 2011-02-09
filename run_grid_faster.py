import numpy as np
import stata_to_python


def stdize(x) :
	return (x-np.mean(x))/np.std(x)



def load_maternal() :
	'''
	Output format: region, country, year, age, y, se, x0...xi-1, w0...wi-1
		Where 	y = ln(death rate)
				se = sampling variance in ln(rate) space
				xi = covariates (and datatype dummies)
				wi = covariate standard error
	'''

	# load in the maternal dataset
	maternal_raw = stata_to_python.genfromdta('/home/j/Project/Causes of Death/CoDMod/pymc-space-time-model/maternal_paper_db_no_missing.dta', missing_flt=np.nan)

	# return the appropriately shaped array
	'''	x0: ln_LDI, x1: ln_LDI_sq, x2: educ, x3: neonatal_mort, x4: tfr, x5: year, x6: age15, x7: age20, x8: age25, x9: age35, x10: age40, x11: age45 '''
	data = zip(maternal_raw['region'],
		maternal_raw['iso3'],
		maternal_raw['year'],
		maternal_raw['age'],
		maternal_raw['ln_rate'],
		maternal_raw['ln_rate_sd'],
		stdize(np.log(maternal_raw['LDI_id'])),
		stdize(np.log(maternal_raw['LDI_id'])**2),
		stdize(maternal_raw['educ']),
		stdize(maternal_raw['q_nn_med']),
		stdize(maternal_raw['tfr']),
		stdize(maternal_raw['year']),
		(maternal_raw['age']==15).astype(float),
		(maternal_raw['age']==20).astype(float),
		(maternal_raw['age']==25).astype(float),
		(maternal_raw['age']==35).astype(float),
		(maternal_raw['age']==40).astype(float),
		(maternal_raw['age']==45).astype(float))

	# filter data for fast debugging and testing
	data = [d for d in data if d[0].startswith('Asia')]
	#data = [d for d in data if (d[2]>=1995)&(d[2]<=2000)]
	#data = [d for d in data if (d[3]>=30)&(d[3]<=40)]
	
	# keep just observed data, we'll build the predictions for out-of-sample later
	data = [d for d in data if np.isnan(d[4])==False]
	
	print 'Data Rows: ', len(data)
	assert len(data) > 0
	
	return np.array(data, dtype=[('region','|S35'),('country','|S3'),('year','<f8'),('age','<f8'),('y','<f8'),('se','<f8'),('x0','<f8'),('x1','<f8'),('x2','<f8'),('x3','<f8'),('x4','<f8'),('x5','<f8'),('x6','<f8'),('x7','<f8'),('x8','<f8'),('x9','<f8'),('x10','<f8'),('x11','<f8')]).view(np.recarray)



# run the model	
import time
print 'started at ', time.localtime()
data = load_maternal()
import codmod_grid_threaded
reload(codmod_grid_threaded)
print('Data loaded', time.localtime())
cm_mod = codmod_grid_threaded.model(data)
print('Model built', time.localtime())
cm_init = codmod_grid_threaded.find_init_vals(cm_mod)
print('MAP found', time.localtime())
cm_init.sample(iter=100, verbose=2)
print('Sampling complete (threaded)', time.localtime())


'''
# save the results
import numpy.lib.recfunctions
predicted_y = cm_init.param_pred.stats()['mean']
results = numpy.lib.recfunctions.append_fields(data, 'prediction', predicted_y)
lower_y = cm_init.param_pred.stats()['95% HPD interval'][:,0]
results = numpy.lib.recfunctions.append_fields(results, 'lower', lower_y)
upper_y = cm_init.param_pred.stats()['95% HPD interval'][:,1]
results = numpy.lib.recfunctions.append_fields(results, 'upper', upper_y)
from pylab import rec2csv
rec2csv(results, '/home/j/Project/Causes of Death/CoDMod/pymc-space-time-model/asia_maternal_results.csv')
'''







