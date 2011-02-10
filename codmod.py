'''
Author:	Kyle Foreman (w/ Abie Flaxman)
Date:	09 February 2011
'''

import pymc as mc
import pymc.gp as gp
import numpy as np
from scipy import interpolate

def model(data, sample_years=[1980.,1990.,2000.,2010.], sample_ages=[15.,25.,35.,45.], year_range=[1980,2010], age_range=[15,45]):
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
    # priors on GP amplitudes
    sigma_r = mc.Exponential('sigma_r', beta=2., value=2.)
    sigma_c = mc.Exponential('sigma_c', beta=1., value=1.)
    # priors on GP scales
    tau_r = mc.Truncnorm('tau_r', mu=15., tau=5.**-2, a=5, b=np.Inf, value=15.)
    tau_c = mc.Truncnorm('tau_c', mu=15., tau=5.**-2, a=5, b=np.Inf, value=15.)

    # find indices for each subset
    regions = np.unique(data.region)
    r_index = [np.where(data.region==r) for r in regions]
    r_list = range(len(regions)_
    countries = np.unique(data.country)
    c_index = [np.where(data.country==c) for c in countries]
    c_list = range(len(countries))
    years = range(year_range[0],year_range[1]+1)
    t_index = dict([(t, i) for i, t in enumerate(years)])
    ages = range(age_range[0],age_range[1]+1,5)
    if age_range[0] == 0:
        ages.insert(1,1)
    elif age_range[0] == 1:
        ages = range(5,age_range[1]+1,5)
        ages.insert(0,1)
    a_index = dict([(a, i) for i, a in enumerate(ages)])
    t_by_r = [[t_index[data.year[j]] for j in r_index[r][0]] for r in r_list]
    a_by_r = [[a_index[data.age[j]] for j in r_index[r][0]] for r in r_list]
    t_by_c = [[t_index[data.year[j]] for j in c_index[c][0]] for c in c_list]
    a_by_c = [[a_index[data.age[j]] for j in c_index[c][0]] for c in c_list]	

    # fixed-effect predictions
    @mc.deterministic
    def fixed_effect(X=X, beta=beta):
        '''fixed_effect_c,t,a = beta * X_c,t,a'''
        return np.dot(beta, X)

    # find all the points on which to evaluate the random effects grid
    sample_points = []
    for a in sample_ages:
        for t in sample_years:
            sample_points.append([a,t])
    sample_points = np.array(sample_points)

    # choose the degree for spline fitting (prefer cubic, but for undersampling pick smaller)
    kx = 3 if len(sample_ages) > 3 else len(sample_ages)-1
    ky = 3 if len(sample_years) > 3 else len(sample_years)-1

    # make variance-covariance matrices for the sampling grid
    @mc.deterministic
    def C_r(s=sample_points, sigma=sigma_r, tau=tau_r):
        return gp.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

    @mc.deterministic
    def C_c(s=sample_points, sigma=sigma_c, tau=tau_c):
        return gp.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

    # draw samples for each random effect matrix
    pi_r_samples = [mc.MvNormalCov('pi_r_%s'%r, np.zeros(sample_points.shape[0]), C_r, value=np.zeros(sample_points.shape[0])) for r in regions]
    pi_c_samples = [mc.MvNormalCov('pi_c_%s'%c, np.zeros(sample_points.shape[0]), C_c, value=np.zeros(sample_points.shape[0])) for c in countries]

    # interpolate to create the complete random effect matrices, then convert into 1d arrays
    @mc.deterministic
    def pi_r(pi_samples=pi_r_samples):
        pi_r = np.zeros(data.shape[0])
        for r in r_list:
            interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[r], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
            pi_r_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
            pi_r[r_index[r]] = pi_r_grid[a_by_r[r],t_by_r[r]]
        return pi_r

    @mc.deterministic
    def pi_c(pi_samples=pi_c_samples):
        pi_c = np.zeros(data.shape[0])
        for c in c_list:
            interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[c], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
            pi_c_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
            pi_c[c_index[c]] = pi_c_grid[a_by_c[c],t_by_c[c]]
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
    '''
    import time as tm
    dbname = '/tmp/codmod_' + str(np.int(tm.time()))
    db = mc.database.pickle.Database(dbname=dbname, dbmode='w')
    '''

    # MCMC step methods
    #mod_mc = mc.MCMC(vars(), db=db)
    mod_mc = mc.MCMC(vars(), db='ram')
    mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.beta)
    
    # use covariance matrix to seed adaptive metropolis steps
    for r in r_list:
        mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01))
    for c in c_list:
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


