'''
Author:	Kyle Foreman (w/ Abie Flaxman)
Date:	04 February 2011
'''

import pymc as mc
import pymc.gp as gp
import numpy as np

def model(data):
	''' Cause of death modelling with random effects correlated over space/time/age

		Y_c,t,a = Beta*X_c,t,a + GP_r(t) + GP_r(a) + GP_c(t) + GP_c(a) + e_c,t,a

			where	r: region
					c: country
					t: year
					a: age

		Y_c,t,a	~ ln(cause-specific death rate)

		Beta 		~ fixed effects (coefficients on covariates)
					  Laplace with Mean = 0
		X			~ covariates, by country/year/age

		GP_r(t)		~ Gaussian Process random effect over time by region
					  Mean = 0; Covariance = Exponential Euclidean over time
		GP_r(a)		~ Gaussian Process random effect over age by region
					  Mean = 0; Covariance = Exponential Euclidean over age
		GP_c(t)		~ Gaussian Process random effect over time by country
					  Mean = 0; Covariance = Exponential Euclidean over time
		GP_c(a)		~ Gaussian Process random effect over age by country
					  Mean = 0; Covariance = Exponential Euclidean over age

		e_c,t,a 	~ Error
					  N(0, sigma_e^2)
	'''

	# make a matrix of covariates (plus an intercept)
	k = len([n for n in data.dtype.names if n.startswith('x')])
	X = np.vstack((np.ones(data.shape[0]),np.array([data['x%d'%i] for i in range(k)])))

	# priors
	beta = mc.Laplace('beta', mu=0., tau=1., value=np.zeros(k+1))
	sigma_e = mc.Exponential('sigma_e', beta=1., value=1.)

	# hyperpriors for GPs
	sigma_f_rt = mc.Exponential('sigma_f_rt', beta=1., value=1.)
	tau_f_rt = mc.Truncnorm('tau_f_rt', mu=15., tau=5.**-2, a=5, b=25, value=15.)
	sigma_f_ra = mc.Exponential('sigma_f_ra', beta=1., value=1.)
	tau_f_ra = mc.Truncnorm('tau_f_ra', mu=15., tau=5.**-2, a=0, b=80, value=15.)
	sigma_f_ct = mc.Exponential('sigma_f_ct', beta=.5, value=.5)
	tau_f_ct = mc.Truncnorm('tau_f_ct', mu=15., tau=5.**-2, a=5, b=25, value=15.)
	sigma_f_ca = mc.Exponential('sigma_f_ca', beta=.5, value=.5)
	tau_f_ca = mc.Truncnorm('tau_f_ca', mu=15., tau=5.**-2, a=0, b=80, value=15.)
	
	# find indices for each subset
	regions = np.unique(data.region)
	r_index = [np.where(data.region==r) for r in regions]
	countries = np.unique(data.country)
	c_index = [np.where(data.country==c) for c in countries]
	years = np.unique(data.year)
	t_index = dict([(t, i) for i, t in enumerate(years)])
	ages = np.unique(data.age)
	a_index = dict([(a, i) for i, a in enumerate(ages)])
	
	# fixed-effect predictions
	@mc.deterministic
	def fixed_effect(X=X, beta=beta):
		'''fixed_effect_c,t,a = beta * X_c,t,a'''
		return np.dot(beta, X)

	# variance-covariance matrices for region GPs
	@mc.deterministic
	def C_rt(sigma_f=sigma_f_rt, tau_f=tau_f_rt, t=years):
		return gp.exponential.euclidean(t, t, amp=sigma_f, scale=tau_f, symm=True)
	@mc.deterministic
	def C_ra(sigma_f=sigma_f_ra, tau_f=tau_f_ra, a=ages):
		return gp.exponential.euclidean(a, a, amp=sigma_f, scale=tau_f, symm=True)
		
	# variance-covariance matrices for country GPs
	@mc.deterministic
	def C_ct(sigma_f=sigma_f_ct, tau_f=tau_f_ct, t=years):
		return gp.exponential.euclidean(t, t, amp=sigma_f, scale=tau_f, symm=True)
	@mc.deterministic
	def C_ca(sigma_f=sigma_f_ca, tau_f=tau_f_ca, a=ages):
		return gp.exponential.euclidean(a, a, amp=sigma_f, scale=tau_f, symm=True)

	# implement GPs as multivariate normals with appropriate covariance structure
	GP_rt = [mc.MvNormalCov('GP_rt_%s'%r, np.zeros_like(years), C_rt, value=np.zeros_like(years)) for r in regions]
	GP_ra = [mc.MvNormalCov('GP_ra_%s'%r, np.zeros_like(ages), C_ra, value=np.zeros_like(ages)) for r in regions]
	GP_ct = [mc.MvNormalCov('GP_ct_%s'%c, np.zeros_like(years), C_ct, value=np.zeros_like(years)) for c in countries]
	GP_ca = [mc.MvNormalCov('GP_ca_%s'%c, np.zeros_like(ages), C_ca, value=np.zeros_like(ages)) for c in countries]
	
	# GP predictions
	@mc.deterministic
	def GP_rt_pred(GP_rt=GP_rt):
		GP_rt_pred = np.zeros(data.shape[0])
		for r in range(len(regions)):
			t = [t_index[data.year[j]] for j in r_index[r][0]]
			GP_rt_pred[r_index[r]] = GP_rt[r][t]
		return GP_rt_pred
	
	@mc.deterministic
	def GP_ra_pred(GP_ra=GP_ra):
		GP_ra_pred = np.zeros(data.shape[0])
		for r in range(len(regions)):
			a = [a_index[data.age[j]] for j in r_index[r][0]]
			GP_ra_pred[r_index[r]] = GP_ra[r][a]
		return GP_ra_pred
	
	@mc.deterministic
	def GP_ct_pred(GP_ct=GP_ct):
		GP_ct_pred = np.zeros(data.shape[0])
		for c in range(len(countries)):
			t = [t_index[data.year[j]] for j in c_index[c][0]]
			GP_ct_pred[c_index[c]] = GP_ct[c][t]
		return GP_ct_pred
	
	@mc.deterministic
	def GP_ca_pred(GP_ca=GP_ca):
		GP_ca_pred = np.zeros(data.shape[0])
		for c in range(len(countries)):
			a = [a_index[data.age[j]] for j in c_index[c][0]]
			GP_ca_pred[c_index[c],:] = GP_ca[c][a]
		return GP_ca_pred

	# parameter predictions
	@mc.deterministic
	def param_pred(fixed_effect=fixed_effect, GP_rt_pred=GP_rt_pred, GP_ra_pred=GP_ra_pred, GP_ct_pred=GP_ct_pred, GP_ca_pred=GP_ca_pred):
		return np.vstack([fixed_effect, GP_rt_pred, GP_ra_pred, GP_ct_pred, GP_ca_pred]).sum(axis=0)

	# data likelihood
	@mc.deterministic
	def tau_pred(sigma_e=sigma_e, var_d=data.se**2.):
		return 1. / (sigma_e**2. + var_d)

	# observe the data
	obs_index = np.where(np.isnan(data.y)==False)
	@mc.observed
	def data_likelihood(value=data.y, i=obs_index, mu=param_pred, tau=tau_pred):
		return mc.normal_like(value[i], mu[i], tau[i])
	
	# create an HDF5 backend to store the model
	import time as tm
	dbname = '/tmp/codmod_' + str(np.int(tm.time()))
	db = mc.database.hdf5.Database(dbname=dbname, dbmode='w')
	
	return mc.MCMC(vars(), db=db)

def fit(model):
	return mc.NormApprox(model).fit()

def sample(norm_approx, n=1000):
	return norm_approx.sample(n)


