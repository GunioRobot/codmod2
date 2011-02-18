'''
Author:	    Kyle Foreman
Date:	    16 February 2011
Purpose:    Fit cause of death models over space, time, and age
'''

import pymc as mc
import numpy as np
import MySQLdb
from scipy import interpolate
from pymc.Matplot import plot as mcplot
import matplotlib as plot
from matplotlib.mlab import rec_join
import numpy.lib.recfunctions as recfunctions

class codmod:
    '''
    codmod has the following methods:

        set_window:         the range of ages/years to predict over
        set_pi_samples:     at what ages/years should pi (the random effect component) be sampled?
        set_covariates:     set which covariates to use in the model
        list_cause_names:   lists the codmod causes available for easy reference
        list_covariates:    lists the covariates available for easy reference
        load:               query the mysql database for the appropriate data
        initialize_model:   create the model object and find starting values via MAP
        sample:             use MCMC to find posterior parameter estimates
        predict:            use the parameter draws to calculate death rate estimates
    '''


    def __init__(self, cause, sex):
        ''' Specify cause (string like 'Aa02', 'Ab10', 'B142', 'C241', etc) and sex ('male' or 'female') '''
        self.connect()
        self.cause = cause
        self.data_rows = 0
        self.sex = sex
        if (self.sex=='male'):
            self.sex_num = 1
        elif (self.sex=='female'):
            self.sex_num = 2
        else:
            raise ValueError("Specify sex as either 'male' or 'female'")
        print 'Sex:', self.sex
        print 'Cause:', self.cause
        self.list_cause_names(cause=self.cause)
        self.set_covariates()
        self.set_window()
        self.set_pi_samples()


    def model_setup(self):
        print 'Cause:', self.cause
        print 'Sex:', self.sex
        print 'Age Range:', self.age_range
        print 'Age Samples:', self.age_samples
        print 'Year Range:', self.year_range
        print 'Year Samples:', self.year_samples
        if self.data_rows == 0:
            print 'Data Rows: Not Loaded'
        else:
            print 'Data Rows:', self.data_rows
        print 'Covariates:', self.covariate_list
        print 'Age Dummies:', self.age_dummies


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


    def list_cause_names(self, cause=''):
        ''' Return a list mapping cause codes with names '''
        if cause == '':
            self.cursor.execute('SELECT cod_cause,cod_cause_name FROM cod_causes WHERE substr(cod_cause,length(cod_cause),1)!="x";')
            return self.cursor.fetchall()
        else:
            self.cursor.execute('SELECT cod_cause_name FROM cod_causes WHERE cod_cause="' + cause + '";')
            self.cause_name = self.cursor.fetchall()[0][0]
            print 'Cause Name:', self.cause_name


    def set_covariates(self, covariate_list=['year','education_yrs_pc'], age_dummies=True, age_ref=30, normalize=True):
        '''
        By default, the model will just use education as a covariate, plus age dummies.
        Calling this method with a list of covariates will set the model to use those instead.
        In addition, some simple transformations (ln(covariate) = natural log, (covariate)^2 = squared) are allowed.
        
        For example, to use ln(LDI) and education as covariates, use this syntax:
            codmod.set_covariates(['education_yrs_pc','ln(LDI_pc)'])
        '''
        if type(covariate_list) == str:
            covariate_list = [covariate_list]
        self.covariate_list = covariate_list
        self.covariates_untransformed = []
        self.covariate_transformations = []
        for c in covariate_list:
            if c[0:3] == 'ln(' and c[-2:] != '^2':
                self.covariates_untransformed.append(c[3:len(c)-1])
                self.covariate_transformations.append('ln')
            elif c[0:3] == 'ln(' and c[-2:] == '^2':
                self.covariates_untransformed.append(c[3:len(c)-3])
                self.covariate_transformations.append('ln+sq')
            elif c[-2:] == '^2':
                self.covariates_untransformed.append(c[:len(c)-3])
                self.covariate_transformations.append('sq')
            else:
                self.covariates_untransformed.append(c)
                self.covariate_transformations.append('none')
        self.age_dummies = age_dummies
        self.age_ref = age_ref
        self.normalize = normalize
        print 'Covariates:', self.covariate_list
        print 'Age Dummies:', self.age_dummies
        print 'Reference Age:', self.age_ref
        print 'Normalize Covariates:', self.normalize


    def load(self):
        '''
        Loads codmod data from the MySQL server.
        The resulting query will get all the data for a specified cause and sex, plus any covariates specified
        '''
        # load in selected covariates
        covs = ''
        for i in list(set(self.covariates_untransformed)):
            if i != 'year':
                covs = covs + i + ','
        cov_sql = 'SELECT iso3 AS country,region,super_region,age,year,sex,' + covs[0:-1] + ' FROM all_covariates WHERE sex=' + str(self.sex_num) + ' AND age BETWEEN ' + str(self.age_range[0]) + ' AND ' + str(self.age_range[1]) + ' AND year BETWEEN ' + str(self.year_range[0]) + ' AND ' + str(self.year_range[1])
        print 'Loading covariates...'
        covariate_data = mysql_to_recarray(self.cursor, cov_sql)        

        # make covariate matrix (including transformations and normalization)
        covariate_vectors = [covariate_data.country, covariate_data.region, covariate_data.super_region, covariate_data.year, covariate_data.age, covariate_data.sex, np.ones(covariate_data.shape[0])]
        covariate_names = ['country', 'region', 'super_region', 'year', 'age', 'sex', 'x0']
        self.covariate_dict = {'x0': 'constant'}
        for i in range(len(self.covariate_list)):
            j = covariate_data[self.covariates_untransformed[i]]
            if self.covariate_transformations[i] == 'ln':
                j = np.log(j)
            elif self.covariate_transformations[i] == 'ln+sq':
                j = (np.log(j))**2
            elif self.covariate_transformations[i] == 'sq':
                j = j**2
            if self.normalize == True:
                j = ((j-np.mean(j))/np.std(j))
            covariate_vectors.append(j)
            covariate_names.append('x' + str(i+1))
            self.covariate_dict['x' + str(i+1)] = self.covariate_list[i]

        # create age dummies if specified
        if self.age_dummies == True:
            pre_ref = 1
            for i,a in enumerate(np.unique(covariate_data.age)):
                if a == self.age_ref:
                    pre_ref = 0
                elif pre_ref == 1:
                    covariate_vectors.append(np.array(covariate_data.age==a).astype(np.float))
                    covariate_names.append('x' + str(len(self.covariate_list)+i+1))
                    self.covariate_dict['x' + str(len(self.covariate_list)+i+1)] = 'Age ' + str(a)
                else:
                    covariate_vectors.append(np.array(covariate_data.age==a).astype(np.float))
                    covariate_names.append('x' + str(len(self.covariate_list)+i))
                    self.covariate_dict['x' + str(len(self.covariate_list)+i)] = 'Age ' + str(a)
        self.covariate_matrix = np.core.records.fromarrays(covariate_vectors, names=covariate_names)

        # load in death observations
        deaths_sql = 'SELECT cf,iso3 AS country,year,sex,age,sample_size,region,envelope,pop FROM full_cod_database WHERE cod_id="' + self.cause + '" AND sex=' + str(self.sex_num) + ' AND age BETWEEN ' + str(self.age_range[0]) + ' AND ' + str(self.age_range[1]) + ' AND year BETWEEN ' + str(self.year_range[0]) + ' AND ' + str(self.year_range[1])
        print 'Loading death data...'
        self.death_obs = mysql_to_recarray(self.cursor, deaths_sql)

        # remove observations in which the CF is missing or not within (0,1)
        self.death_obs = np.delete(self.death_obs, np.where((np.isnan(self.death_obs.cf)) | (self.death_obs.cf > 1) | (self.death_obs.cf < 0))[0], axis=0)

        # set sample size to 10 when sample size is missing
        self.death_obs.sample_size[np.where((np.isnan(self.death_obs.sample_size)) | (self.death_obs.sample_size < 10.))] = 10.

        # apply a moving average (5 year window) on cause fractions of 0 or 1, or where sample size is less than 100
        country_age_lookups = {}
        for c in np.unique(self.death_obs.country):
            for a in np.unique(self.death_obs.age):
                country_age_lookups[c+str(a)] = np.where((self.death_obs.age == a) & (self.death_obs.country == c))[0]
        year_window_lookups = {}
        for y in range(self.year_range[0],self.year_range[1]+1):
            year_window_lookups[y] = np.where((self.death_obs.year >= y-2.) & (self.death_obs.year <= y+2.))[0]
        smooth_me = np.where((self.death_obs.cf==0.) | (self.death_obs.cf==1.) | (self.death_obs.sample_size<100.))[0]
        for i in smooth_me:
            self.death_obs.cf[i] = self.death_obs.cf[np.intersect1d(country_age_lookups[self.death_obs.country[i]+str(self.death_obs.age[i])],year_window_lookups[self.death_obs.year[i]])].mean()

        # for cases in which the CF is still 0 or 1 after the moving average, use the smallest/largest non-0/1 CF observed in that region-age
        region_age_lookups = {}
        for r in np.unique(self.death_obs.region):
            for a in np.unique(self.death_obs.age):
                region_age_lookups[str(r)+'_'+str(a)] = np.where((self.death_obs.age == a) & (self.death_obs.region == r))[0]
        nonzeros = np.where(self.death_obs.cf>0)[0]
        nonones = np.where(self.death_obs.cf<1)[0]
        for i in np.where(self.death_obs.cf==0.)[0]:
            candidates = np.intersect1d(region_age_lookups[str(self.death_obs.region[i])+'_'+str(self.death_obs.age[i])], nonzeros)
            if candidates.shape[0] == 0:
                self.death_obs.cf[i] = 0.
            else:
                self.death_obs.cf[i] = self.death_obs.cf[candidates].min()
        for i in np.where(self.death_obs.cf==1.)[0]:
            candidates = np.intersect1d(region_age_lookups[str(self.death_obs.region[i])+'_'+str(self.death_obs.age[i])], nonones)
            if candidates.shape[0] == 0:
                self.death_obs.cf[i] = 1.
            else:
                self.death_obs.cf[i] = self.death_obs.cf[candidates].max()

        # finally, any CF that is still 0 or 1 after the above corrections should simply be dropped
        self.death_obs = np.delete(self.death_obs, np.where((self.death_obs.cf == 0.) | (self.death_obs.cf == 1.))[0], axis=0)

        # y is the observed number of deaths
        y = self.death_obs.cf * self.death_obs.sample_size
        obs = np.core.records.fromarrays([self.death_obs.country, self.death_obs.year, self.death_obs.age, self.death_obs.sex, y, self.death_obs.envelope, self.death_obs.pop, self.death_obs.sample_size], names=['country','year','age','sex','y','envelope','pop','sample_size'])

        # prep all the in-sample data
        self.training_data = rec_join(['country','year','age','sex'], obs, self.covariate_matrix)
        self.data_rows = self.training_data.shape[0]
        print 'Data Rows:', self.data_rows


    def plot_data(self, country=''):
        if country:
            return something


    def initialize_model(self, find_start_vals=True):
        '''
        Y_c,t,a ~ Negative Binomial(mu_c,t,a, alpha)
        
            where	s: super-region
                    r: region
                    c: country
                    t: year
                    a: age

            Y_c,t,a		~ observed deaths due to a cause in a country/year/age/sex
            
            mu_c,t,a    ~ exp(beta*X_c,t,a + ln(E) + pi_s + pi_r + pi_c + e_c,t,a)
            
                        beta    ~ fixed effects (coefficients on covariates)
                                  Laplace with Mean = 0
                        X_c,t,a ~ covariates (by country/year/age)
                        
                        E       ~ exposure (total number of all-cause deaths observed)
                        
                        pi_s    ~ 'random effect' by super-region
                                  year*age grid of offsets
                                  sampled from MVN with matern covariance then interpolated via cubic spline
                        pi_r    ~ 'random effect' by region
                                  year*age grid of offsets
                                  sampled from MVN with matern covariance then interpolated via cubic spline
                        pi_c    ~ 'random effect' by country
                                  year*age grid of offsets
                                  sampled from MVN with matern covariance then interpolated via cubic spline

                        e_c,t,a ~ error
                        
            alpha       ~ overdispersion parameter
        '''
        # make a matrix of covariates
        k = len([n for n in self.training_data.dtype.names if n.startswith('x')])
        X = np.array([self.training_data['x%d'%i] for i in range(k)])

        # prior on beta (covariate coefficients)
        beta = mc.Laplace('beta', mu=0.0, tau=1.0, value=np.zeros(k))
        # prior on alpha (overdispersion parameter)
        alpha = mc.Exponential('alpha', beta=1.0, value=1.0)
        # priors on matern amplitudes
        sigma_s = mc.Exponential('sigma_s', beta=2.0, value=2.0)
        sigma_r = mc.Exponential('sigma_r', beta=1.5, value=1.5)
        sigma_c = mc.Exponential('sigma_c', beta=1.0, value=1.0)
        # priors on matern scales
        tau_s = mc.Truncnorm('tau_s', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)
        tau_r = mc.Truncnorm('tau_r', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)
        tau_c = mc.Truncnorm('tau_c', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)

        # find indices for each subset
        super_regions = np.unique(self.training_data.super_region)
        s_index = [np.where(self.training_data.super_region==s) for s in super_regions]
        s_list = range(len(super_regions))
        regions = np.unique(self.training_data.region)
        r_index = [np.where(self.training_data.region==r) for r in regions]
        r_list = range(len(regions))
        countries = np.unique(self.training_data.country)
        c_index = [np.where(self.training_data.country==c) for c in countries]
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
        t_by_s = [[t_index[self.training_data.year[j]] for j in s_index[s][0]] for s in s_list]
        a_by_s = [[a_index[self.training_data.age[j]] for j in s_index[s][0]] for s in s_list]
        t_by_r = [[t_index[self.training_data.year[j]] for j in r_index[r][0]] for r in r_list]
        a_by_r = [[a_index[self.training_data.age[j]] for j in r_index[r][0]] for r in r_list]
        t_by_c = [[t_index[self.training_data.year[j]] for j in c_index[c][0]] for c in c_list]
        a_by_c = [[a_index[self.training_data.age[j]] for j in c_index[c][0]] for c in c_list]	

        # fixed-effect predictions
        @mc.deterministic
        def fixed_effect(X=X, beta=beta):
            '''fixed_effect_c,t,a = beta * X_c,t,a'''
            return np.dot(beta, X)

        # find all the points on which to evaluate the random effects grid
        sample_points = []
        for a in self.age_samples:
            for t in self.year_samples:
                sample_points.append([a,t])
        sample_points = np.array(sample_points)

        # choose the degree for spline fitting (prefer cubic, but for undersampling pick smaller)
        kx = 3 if len(self.age_samples) > 3 else len(self.age_samples)-1
        ky = 3 if len(self.year_samples) > 3 else len(self.year_samples)-1

        # make variance-covariance matrices for the sampling grid
        @mc.deterministic
        def C_s(s=sample_points, sigma=sigma_s, tau=tau_s):
            return mc.gp.cov_funs.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

        @mc.deterministic
        def C_r(s=sample_points, sigma=sigma_r, tau=tau_r):
            return mc.gp.cov_funs.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

        @mc.deterministic
        def C_c(s=sample_points, sigma=sigma_c, tau=tau_c):
            return mc.gp.cov_funs.matern.euclidean(s, s, amp=sigma, scale=tau, diff_degree=2., symm=True)

        # draw samples for each random effect matrix
        pi_s_samples = [mc.MvNormalCov('pi_s_%s'%s, np.zeros(sample_points.shape[0]), C_s, value=np.zeros(sample_points.shape[0])) for s in super_regions]
        pi_r_samples = [mc.MvNormalCov('pi_r_%s'%r, np.zeros(sample_points.shape[0]), C_r, value=np.zeros(sample_points.shape[0])) for r in regions]
        pi_c_samples = [mc.MvNormalCov('pi_c_%s'%c, np.zeros(sample_points.shape[0]), C_c, value=np.zeros(sample_points.shape[0])) for c in countries]

        # interpolate to create the complete random effect matrices, then convert into 1d arrays
        @mc.deterministic
        def pi_s(pi_samples=pi_s_samples):
            pi_s = np.zeros(self.training_data.shape[0])
            for s in s_list:
                interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[s], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
                pi_s_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
                pi_s[s_index[s]] = pi_s_grid[a_by_s[s],t_by_s[s]]
            return pi_s

        @mc.deterministic
        def pi_r(pi_samples=pi_r_samples):
            pi_r = np.zeros(self.training_data.shape[0])
            for r in r_list:
                interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[r], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
                pi_r_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
                pi_r[r_index[r]] = pi_r_grid[a_by_r[r],t_by_r[r]]
            return pi_r

        @mc.deterministic
        def pi_c(pi_samples=pi_c_samples):
            pi_c = np.zeros(self.training_data.shape[0])
            for c in c_list:
                interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples[c], xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
                pi_c_grid = interpolate.bisplev(x=ages, y=years, tck=interpolator)
                pi_c[c_index[c]] = pi_c_grid[a_by_c[c],t_by_c[c]]
            return pi_c

        # parameter predictions
        @mc.deterministic
        def param_pred(fixed_effect=fixed_effect, pi_s=pi_s, pi_r=pi_r, pi_c=pi_c, E=self.training_data.sample_size):
            return np.exp(np.vstack([fixed_effect, np.log(E), pi_s, pi_r, pi_c]).sum(axis=0))

        # observe the data
        @mc.observed
        def data_likelihood(value=self.training_data.y, mu=param_pred, alpha=alpha):
            return mc.negative_binomial_like(value, mu, alpha)

        # create a pickle backend to store the model
        import time as tm
        dbname = '/home/j/Project/Causes of Death/CoDMod/tmp files/codmod_' + self.cause + '_' + tm.strftime('%b%d_%I%M%p')
        db = mc.database.pickle.Database(dbname=dbname, dbmode='w')

        # MCMC step methods
        self.mod_mc = mc.MCMC(vars(), db=db)
        #self.mod_mc = mc.MCMC(vars(), db='ram')
        self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.beta)
        
        # use covariance matrix to seed adaptive metropolis steps
        for s in s_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_s_samples[s], cov=np.array(C_s.value*.01))
        for r in r_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01))
        for c in c_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_c_samples[c], cov=np.array(C_c.value*.01))
        
        # find good initial conditions with MAP approximation
        for var_list in [[self.mod_mc.data_likelihood, self.mod_mc.beta]] + \
            [[self.mod_mc.data_likelihood, s] for s in self.mod_mc.pi_s_samples] + \
            [[self.mod_mc.data_likelihood, r] for r in self.mod_mc.pi_r_samples] + \
            [[self.mod_mc.data_likelihood, c] for c in self.mod_mc.pi_c_samples] + \
            [[self.mod_mc.data_likelihood, self.mod_mc.beta]]:
            print 'attempting to maximize likelihood of %s' % [v.__name__ for v in var_list]
            mc.MAP(var_list).fit(method='fmin_powell', verbose=1)
            print ''.join(['%s: %s\n' % (v.__name__, v.value) for v in var_list[1:]])


    def sample(self, iter=5000, burn=1000, thin=5, verbose=1, chains=1):
        ''' Use MCMC to sample from the posterior '''
        self.mod_mc.sample(iter=iter, burn=burn, thin=thin, verbose=verbose)


def mysql_to_recarray(cursor, query):
    ''' Makes a MySQL query and returns the results as a record array '''
    cursor.execute(query)
    data = cursor.fetchall()
    data = [data[i] for i in range(len(data))]
    cols = np.array([cursor.description[i][0:2] for i in range(len(cursor.description))])
    for i in range(len(cols)) :
        t = cols[i][1]
        if t == '1' or t == '2' or t == '9' or t == '13':
            cols[i][1] = '<i4'
        elif t == '4' or t == '5' or t =='8' or t == '3':
            cols[i][1] = '<f8'
        else:
            str_l = 1
            for j in range(len(data)) :
                str_l = max((str_l, len(data[j][i])))
            cols[i][1] = '<S' + str(str_l)
    cols = [(cols[i][0], cols[i][1]) for i in range(len(cols))]
    return np.array(data, dtype=cols).view(np.recarray)

