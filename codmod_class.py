'''
Author:	    Kyle Foreman
Date:	    09 February 2011
Purpose:    Fit cause of death models over space, time, and age
'''

import pymc as mc
import numpy as np
import MySQLdb
from scipy import interpolate
import matplotlib as plot

class codmod:
    '''
    codmod has the following methods:

        set_window:         the range of ages/years to predict over
        set_pi_samples:     at what ages/years should pi (the random effect component) be sampled?
        set_covariates:     set which covariates to use in the model
        list_causes:        lists the codmod causes available for easy reference
        list_covariates:    lists the covariates available for easy reference
        load:               query the mysql database for the appropriate data
        initialize_model:   create the model object and find starting values via MAP
        sample:             use MCMC to find posterior parameter estimates
        predict:            use the parameter draws to calculate death rate estimates
    '''


    def __init__(self, cause, sex):
        ''' Specify cause (string like 'Aa02', 'Ab10', 'B142', 'C241', etc) and sex ('male' or 'female') '''
        self.cause = cause
        if (sex=='male'):
            self.sex = 1
        elif (sex=='female'):
            self.sex = 2
        else:
            raise ValueError("Specify sex as either 'male' or 'female'")
        self.connect()
        self.set_covariates()
        self.set_window()
        self.set_pi_samples()


    def set_window(self, age_range=[0,80], year_range=[1980,2010]):
        ''' Change which year and age ranges the model predicts for '''
        self.age_range = age_range
        self.year_range = year_range
        print 'Age Range:', age_range
        print 'Year Range:', year_range


    def connect(self):
        '''
        Connect to the MySQL database.
        There should be a file .mysql.cnf in the same directory, formatted as such:
            host = 'concrete.ihme.washington.edu'
            db = 'codmod'
            user = 'codmod'
            passwd = 'password'
        '''
        mysql_opts = open('./mysql.cnf')
        for l in mysql_opts:
            exec l
        self.mysql = MySQLdb.connect(host=host, db=db, user=user, passwd=passwd)
        self.cursor = self.mysql.cursor()
        self.dcursor = self.mysql.cursor(cursorclass=MySQLdb.cursors.DictCursor)


    def set_pi_samples(self, age_samples=[0,1,15,25,40,55,65,80], year_samples=[1980,1990,2000,2010]):
        ''' Change which years and ages to sample pi (the random effect component) at '''
        self.age_samples = age_samples
        self.year_samples = year_samples
        print 'Age Samples:', age_samples
        print 'Year Samples:', year_samples


    def list_covariates(self):
        ''' Return a list of which covariates are available for use in the model '''
        self.cursor.execute('SELECT variable_name,variable_label FROM covariate_list;')
        return self.cursor.fetchall()


    def list_causes(self):
        ''' Return a list mapping cause codes with names '''
        self.cursor.execute("SELECT cod_cause,cod_cause_name FROM cod_causes WHERE substr(cod_cause,length(cod_cause),1)!='x';")
        return self.cursor.fetchall()


    def set_covariates(self, covariate_list=['education_years_pc'], age_dummies=True):
        '''
        By default, the model will just use education as a covariate, plus age dummies.
        Calling this method with a list of covariates will set the model to use those instead.
        In addition, some simple transformations (ln() = natural log) are allowed.
        
        For example, to use ln(LDI) and education as covariates, use this syntax:
            codmod.set_covariates(['education_yrs_pc','ln(LDI_pc)'])
        '''
        self.covariate_list = covariate_list
        self.covariates_untransformed = []
        self.covariate_transformations = []
        for c in covariate_list:
            if c[0:3] == 'ln(':
                self.covariates_untransformed.append(c[3:len(c)-1])
                self.covariate_transformations.append('ln')
            else:
                self.covariates_untransformed.append(c)
                self.covariate_transformations.append('')
        self.age_dummies = age_dummies
        print 'Covariates:', covariate_list
        print 'Age Dummies:', age_dummies


    def load(self):
        '''
        Loads codmod data from the MySQL server.
        The resulting query will get all the data for a specified cause and sex, plus any covariates specified
        '''
        sql = 'SELECT '


    def plot_data(self):
        return something

    def initialize_model(self, find_start_vals=True):
        '''
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
                      calculated by sampling a few year/age pairs then interpolating via cubic spline
        pi_c		~ 'random effect' by country
                      a year*age grid of offsets
                      calculated by sampling a few year/age pairs then interpolating

        e_c,t,a 	~ Error
                      N(0, sigma_e^2)
        '''
        # make a matrix of covariates (plus an intercept)
        k = len([n for n in self.data.dtype.names if n.startswith('x')])
        X = np.vstack((np.ones(self.data.shape[0]),np.array([self.data['x%d'%i] for i in range(k)])))

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
        regions = np.unique(self.data.region)
        r_index = [np.where(self.data.region==r) for r in regions]
        r_list = range(len(regions))
        countries = np.unique(self.data.country)
        c_index = [np.where(self.data.country==c) for c in countries]
        c_list = range(len(countries))
        years = range(self.year_range[0],self.year_range[1]+1)
        t_index = dict([(t, i) for i, t in enumerate(years)])
        ages = range(self.age_range[0],self.age_range[1]+1,5)
        if self.age_range[0] == 0:
            ages.insert(1,1)
        elif self.age_range[0] == 1:
            ages = range(5,self.age_range[1]+1,5)
            ages.insert(0,1)
        a_index = dict([(a, i) for i, a in enumerate(ages)])
        t_by_r = [[t_index[self.data.year[j]] for j in r_index[r][0]] for r in r_list]
        a_by_r = [[a_index[self.data.age[j]] for j in r_index[r][0]] for r in r_list]
        t_by_c = [[t_index[self.data.year[j]] for j in c_index[c][0]] for c in c_list]
        a_by_c = [[a_index[self.data.age[j]] for j in c_index[c][0]] for c in c_list]	

        # fixed-effect predictions
        @mc.deterministic
        def fixed_effect(X=X, beta=beta):
            '''fixed_effect_c,t,a = beta * X_c,t,a'''
            return np.dot(beta, X)

        # find all the points on which to evaluate the random effects grid
        sample_points = []
        for a in self.sample_ages:
            for t in self.sample_years:
                sample_points.append([a,t])
        sample_points = np.array(sample_points)

        # choose the degree for spline fitting (prefer cubic, but for undersampling pick smaller)
        kx = 3 if len(self.sample_ages) > 3 else len(self.sample_ages)-1
        ky = 3 if len(self.sample_years) > 3 else len(self.sample_years)-1

        # make variance-covariance matrices for the sampling grid
        @mc.deterministic
        def C_r(s=sample_points, sigma=sigma_r, tau=tau_r):
            return mc.gp.cov_funs.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

        @mc.deterministic
        def C_c(s=sample_points, sigma=sigma_c, tau=tau_c):
            return mc.gp.cov_funs.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

        # draw samples for each random effect matrix
        pi_r_samples = [mc.MvNormalCov('pi_r_%s'%r, np.zeros(sample_points.shape[0]), C_r, value=np.zeros(sample_points.shape[0])) for r in regions]
        pi_c_samples = [mc.MvNormalCov('pi_c_%s'%c, np.zeros(sample_points.shape[0]), C_c, value=np.zeros(sample_points.shape[0])) for c in countries]

        # interpolate to create the complete random effect matrices, then convert into 1d arrays
        @mc.deterministic
        def pi_r(pi_samples=pi_r_samples):
            pi_r = np.zeros(self.data.shape[0])
            for r in r_list:
                interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[r], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
                pi_r_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
                pi_r[r_index[r]] = pi_r_grid[a_by_r[r],t_by_r[r]]
            return pi_r

        @mc.deterministic
        def pi_c(pi_samples=pi_c_samples):
            pi_c = np.zeros(self.data.shape[0])
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
        def tau_pred(sigma_e=sigma_e, var_d=self.data.se**2.):
            return 1. / (sigma_e**2. + var_d)

        # observe the data
        obs_index = np.where(np.isnan(self.data.y)==False)
        @mc.observed
        def data_likelihood(value=self.data.y, i=obs_index, mu=param_pred, tau=tau_pred):
            return mc.normal_like(value[i], mu[i], tau[i])
        
        # create a pickle backend to store the model
        '''
        import time as tm
        dbname = '/home/j/Project/Causes of Death/CoDMod/tmp files/codmod_' + str(np.int(tm.time()))
        db = mc.database.pickle.Database(dbname=dbname, dbmode='w')
        '''

        # MCMC step methods
        self.mod_mc = mc.MCMC(vars(), db=db)
        #self.mod_mc = mc.MCMC(vars(), db='ram')
        self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.beta)
        
        # use covariance matrix to seed adaptive metropolis steps
        for r in r_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01))
        for c in c_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_c_samples[c], cov=np.array(C_c.value*.01))
        
        # find good initial conditions with MAP approximation
        for var_list in [[self.mod_mc.data_likelihood, self.mod_mc.beta, self.mod_mc.sigma_e]] + \
            [[self.mod_mc.data_likelihood, r] for r in self.mod_mc.pi_r_samples] + \
            [[self.mod_mc.data_likelihood, c] for c in self.mod_mc.pi_c_samples] + \
            [[self.mod_mc.data_likelihood, self.mod_mc.beta, self.mod_mc.sigma_e]]:
            print 'attempting to maximize likelihood of %s' % [v.__name__ for v in var_list]
            mc.MAP(var_list).fit(method='fmin_powell', verbose=1)
            print ''.join(['%s: %s\n' % (v.__name__, v.value) for v in var_list[1:]])


    def sample(self, iter=5000, burn=1000, thin=5, verbose=1, chains=1):
        ''' Use MCMC to sample from the posterior '''
        self.mod_mc.sample(iter=iter, burn=burn, thin=thin, verbose=verbose)



