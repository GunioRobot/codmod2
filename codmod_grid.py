'''
Author:	Kyle Foreman (w/ Abie Flaxman)
Date:	04 February 2011
'''

import pymc as mc
import pymc.gp as gp
import numpy as np
from scipy import interpolate

def model(data, sample_years=[1980.,1990.,2000.,2010.], sample_ages=[15.,25.,35.,45.], year_range=[1980,2010]):
	''' Cause of death modelling with random effects correlated over space/time/age

		Y_c,t,a = beta*X_c,t,a + pi_r + pi_c + e_c,t,a

			where	r: region
					c: country
					t: year
					a: age

		Y_c,t,a		~ ln(cause-specific death rate)

		beta 		~ fixed effects (coefficients on covariates)
					  Laplace with Mean = 0
		X			~ covariates, by country/year/age

		pi_r		~ 'random effect' by region
					  a year*age grid of offsets
					  calculated by sampling a few year/age pairs then interpolating
		pi_c		~ 'random effect' by country
					  a year*age grid of offsets
					  calculated by sampling a few year/age pairs then interpolating

		e_c,t,a 	~ Error
					  N(0, sigma_e^2)
	'''

	# make a matrix of covariates (plus an intercept)
	k = len([n for n in data.dtype.names if n.startswith('x')])
	X = np.vstack((np.ones(data.shape[0]),np.array([data['x%d'%i] for i in range(k)])))

	# prior on beta (covariate coefficients)
	beta = mc.Laplace('beta', mu=0., tau=1., value=np.zeros(k+1))
	# prior on sd of error term
	sigma_e = mc.Exponential('sigma_e', beta=1., value=1.)
	# priors on sd of grid samples
	tau_r = mc.Gamma('tau_r', alpha=2., beta=2., value=1.)
	tau_c = mc.Gamma('tau_c', alpha=2., beta=1., value=2.)
	# priors on covariance multiplier (relationship between points on grid)
	rho_r = mc.Truncnorm('rho_r', mu=1., tau=1., a=0., b=3., value=1.)
	rho_c = mc.Truncnorm('rho_c', mu=1., tau=1., a=0., b=5., value=1.)

	# find indices for each subset
	regions = np.unique(data.region)
	r_index = [np.where(data.region==r) for r in regions]
	countries = np.unique(data.country)
	c_index = [np.where(data.country==c) for c in countries]
	years = range(year_range[0],year_range[1]+1)
	t_index = dict([(t, i) for i, t in enumerate(years)])
	ages = np.unique(data.age)
	a_index = dict([(a, i) for i, a in enumerate(ages)])
	
	# fixed-effect predictions
	@mc.deterministic
	def fixed_effect(X=X, beta=beta):
		'''fixed_effect_c,t,a = beta * X_c,t,a'''
		return np.dot(beta, X)
	
	# find grid distances
	C_grid = np.zeros((len(sample_years)*len(sample_ages), len(sample_years)*len(sample_ages)))
	for x1 in range(len(sample_years)):
		for x2 in range(len(sample_ages)):
			for y1 in range(len(sample_years)):
				for y2 in range(len(sample_ages)):
					C_grid[x1*len(sample_ages)+x2,y1*len(sample_ages)+y2] = (1.-np.abs(np.float(x1)-np.float(y1))/(len(sample_years)+1.))*(1.-np.abs(np.float(x2)-np.float(y2))/(len(sample_ages)+1.))

	# figure out where each sample point falls in the overall grid
	a_grid_lookup = np.empty((len(sample_years)*len(sample_ages)))
	t_grid_lookup = np.empty((len(sample_years)*len(sample_ages)))
	for x1,t in enumerate(sample_years):
		for x2,a in enumerate(sample_ages):
			a_grid_lookup[x1*len(sample_ages)+x2] = a_index[a]
			t_grid_lookup[x1*len(sample_ages)+x2] = t_index[t]

	# make variance-covariance matrices for the sampling grid
	@mc.deterministic
	def C_r(tau=tau_r, rho=rho_r, C_grid=C_grid):
		C = C_grid * rho
		for i in range(C.shape[0]):
			C[i,i] = tau
		return C

	@mc.deterministic
	def C_c(tau=tau_c, rho=rho_c, C_grid=C_grid):
		C = C_grid * rho
		for i in range(C.shape[0]):
			C[i,i] = tau
		return C

	# draw samples for each random effect matrix
	pi_r_samples = [mc.MvNormalCov('pi_r_%s'%r, np.zeros(C_grid.shape[0]), C_r, value=np.zeros(C_grid.shape[0])) for r in regions]
	pi_c_samples = [mc.MvNormalCov('pi_c_%s'%c, np.zeros(C_grid.shape[0]), C_c, value=np.zeros(C_grid.shape[0])) for c in countries]

	# interpolate to create the complete random effect matrices, then convert into 1d arrays
	@mc.deterministic
	def pi_r(pi_samples=pi_r_samples):
		pi_r = np.zeros(data.shape[0])
		for r in range(len(regions)):
			interpolator = interpolate.interp2d(x=a_grid_lookup, y=t_grid_lookup, z=pi_samples[r], kind='cubic', bounds_error=False, fill_value=0.)
			pi_r_grid = interpolator(x=ages, y=years)
			t = [t_index[data.year[j]] for j in r_index[r][0]]
			a = [a_index[data.age[j]] for j in r_index[r][0]]
			pi_r[r_index[r]] = pi_r_grid[t,a]
		return pi_r

	@mc.deterministic
	def pi_c(pi_samples=pi_c_samples):
		pi_c = np.zeros(data.shape[0])
		for c in range(len(countries)):
			interpolator = interpolate.interp2d(x=a_grid_lookup, y=t_grid_lookup, z=pi_samples[c], kind='cubic', bounds_error=False, fill_value=0.)
			pi_c_grid = interpolator(x=ages, y=years)
			t = [t_index[data.year[j]] for j in c_index[c][0]]
			a = [a_index[data.age[j]] for j in c_index[c][0]]
			pi_c[c_index[c]] = pi_c_grid[t,a]		
		return pi_c

	# parameter predictions
	@mc.deterministic
	def param_pred(fixed_effect=fixed_effect, pi_r=pi_r, pi_c=pi_c):
		return np.vstack([fixed_effect, pi_r, pi_c]).sum(axis=0)

	# data likelihood
	@mc.deterministic
	def tau_pred(sigma_e=sigma_e, var_d=data.se**2.):
		return 1. / (sigma_e**2. + var_d)

	# observe the data
	obs_index = np.where(np.isnan(data.y)==False)
	@mc.observed
	def data_likelihood(value=data.y, i=obs_index, mu=param_pred, tau=tau_pred):
		return mc.normal_like(value[i], mu[i], tau[i])
	
	# create a pickle backend to store the model
	import time as tm
	dbname = '/tmp/codmod_' + str(np.int(tm.time()))
	db = mc.database.pickle.Database(dbname=dbname, dbmode='w')

	# MCMC step methods
	mod_mc = mc.MCMC(vars(), db=db)
	mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.beta)
	
	# use covariance matrix to seed adaptive metropolis steps
	for r in range(len(regions)):
		mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01))
	for c in range(len(countries)):
		mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_c_samples[c], cov=np.array(C_c.value*.01))
	
	# return the whole object as a model
	return mod_mc

def find_init_vals(mod_mc):
	# find good initial conditions with MAP approximation
	for var_list in [[mod_mc.data_likelihood, mod_mc.beta, mod_mc.sigma_e]] + \
		[[mod_mc.data_likelihood, r] for r in mod_mc.pi_r_samples] + \
		[[mod_mc.data_likelihood, c] for c in mod_mc.pi_c_samples] + \
		[[mod_mc.data_likelihood, mod_mc.beta, mod_mc.sigma_e]]:
		print 'attempting to maximize likelihood of %s' % [v.__name__ for v in var_list]
		mc.MAP(var_list).fit(method='fmin_powell', verbose=1)
		print ''.join(['%s: %s\n' % (v.__name__, v.value) for v in var_list[1:]])
	return mod_mc

def sample(mod_mc, n=1000):
	mod_mc.sample(n)
	return mod_mc


