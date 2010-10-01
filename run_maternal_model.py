def load_maternal() :
	'''
	Output format: region, country, year, age, y, se, x0...xi-1, w0...wi-1
		Where 	y = ln(death rate)
				se = sampling variance in ln(rate) space
				xi = covariates (and datatype dummies)
				wi = covariate standard error
	'''
	'''
	# connect to the mysql database (must have ~/.my.cnf containing user/pw)
	import MySQLdb
	db = MySQLdb.connect(host='140.142.16.74', port=2302, read_default_file="~/.my.cnf", db='codmod', use_unicode=True)

	# construct the query
	# covariates in paper: ln_tfr, ln_LDI_id, q_nn_med, educ, hiv_5lag, hiv_5lag_sq
	sql = 'SELECT region, iso3, year, age, cf, envelope, population, sample_size, source_type, tfr, education, hiv_5lag, LDI_id, neonatal_mortality_rate FROM codmod_database WHERE cause="A10" AND sex=2 AND age>=15 AND age<=49 AND year>=1980;'

	# execute the query on the database
	c = db.cursor()
	c.execute(sql)

	# save the results into an array
	data = c.fetchall()
	data = [data[i] for i in range(len(data))]

	# hmmmmm..... need to figure out how to get out of sample in there as well...
	'''

	# import tool to load stata data into python
	run '/home/j/Project/Causes of Death/CoDMod/Database/Code/stata_to_python.py'

	# load in the maternal dataset
	import numpy
	maternal_raw = genfromdta('/home/j/Project/Causes of Death/CoDMod/Archive/CODMOD/Maternal Reviewer Responses/Final Run 3/maternal_db_from_paper_for_python.dta', missing_flt=numpy.NaN)

	# create dummy variables by age/region
	age_dummies = (maternal_raw['age'][:, None] == numpy.unique(maternal_raw['age'])).astype(float)
	age_dummies.dtype = [('age_' + str(a.__int__()), 'float64') for a in numpy.unique(maternal_raw['age'])]
	
	
	
	maternal_raw = numpy.concatenate((maternal_raw, age_dummies), axis=1)


	# return the appropriately shaped array
	'''
	x0: ln_LDI, x1: ln_LDI_sq, x2: educ, x3: neonatal_mort, x4: tfr, x5: year
	'''
	return numpy.array(zip(maternal_raw['region'], maternal_raw['country'], (maternal_raw['year']-numpy.mean(maternal_raw['year']))/numpy.std(maternal_raw['year']), (maternal_raw['age']-numpy.mean(maternal_raw['age']))/numpy.std(maternal_raw['age']), maternal_raw['ln_rate'], maternal_raw['ln_rate_sd'], numpy.log(maternal_raw['LDI_id']), numpy.log(maternal_raw['LDI_id'])**2, maternal_raw['educ'], maternal_raw['q_nn_med'], maternal_raw['tfr'], maternal_raw['year'], numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw)), numpy.zeros(len(maternal_raw))), dtype=[('region','|S35'),('country','|S3'),('year','<f4'),('age','<f4'),('y','<f4'),('se','<f4'),('x0','<f4'),('x1','<f4'),('x2','<f4'),('x3','<f4'),('x4','<f4'),('x5','<f4'),('w0','<f4'),('w1','<f4'),('w2','<f4'),('w3','<f4'),('w4','<f4'),('w5','<f4')])

data = load_maternal()
def fit_maternal(data=data) :
	# load the model
	run '/home/j/Project/Causes of Death/CoDMod/pymc-space-time-model/model.py'

	# 














