import MySQLdb

def load_maternal() :
	'''
	Connect to the mysql database, grab maternal deaths and covariates
	
	Output format: region, country, year, age, y, se, x0...xi-1, w0...wi-1
		Where 	y = ln(death rate)
				se = sampling variance in ln(rate) space
				xi = covariates (and datatype dummies)
				wi = covariate standard error
	'''
	
	# connect to the mysql database (must have ~/.my.cnf containing user/pw)
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