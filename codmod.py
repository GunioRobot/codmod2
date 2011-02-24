'''
Author:	    Kyle Foreman
Date:	    22 February 2011
Purpose:    Fit cause of death models over space, time, and age
'''

import pymc as mc
import numpy as np
import MySQLdb
from scipy import interpolate
from pymc.Matplot import plot as mcplot
import matplotlib as plot
import numpy.lib.recfunctions as recfunctions
import gradient_samplers as gs

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
        # make the sql covariate query
        covs = ''
        for i in list(set(self.covariates_untransformed)):
            if i != 'year':
                covs = covs + i + ', '
        covs = covs[0:-2]

        # load observed deaths plus covariates
        obs_sql = 'SELECT iso3 as country, a.region, a.super_region, age, year, sex, cf, sample_size, a.envelope, a.pop, ' + covs + ' FROM full_cod_database AS a LEFT JOIN all_covariates USING (iso3,year,sex,age) WHERE a.cod_id="' + self.cause + '";'
        obs = mysql_to_recarray(self.cursor, obs_sql)
        obs = obs[np.where((obs.year >= self.year_range[0]) & (obs.year <= self.year_range[1]) & (obs.age >= self.age_range[0]) & (obs.age <= self.age_range[1]) & (obs.sex == self.sex_num))[0]]

        # load in just covariates (for making predictions)
        all_sql = 'SELECT iso3 as country, region, super_region, age, year, sex, envelope, pop, ' + covs + ' FROM all_covariates;'
        all = mysql_to_recarray(self.cursor, all_sql)
        all = all[np.where((all.year >= self.year_range[0]) & (all.year <= self.year_range[1]) & (all.age >= self.age_range[0]) & (all.age <= self.age_range[1]) & (all.sex == self.sex_num))[0]]
        
        # get rid of rows for which covariates are unavailable
        for i in list(set(self.covariates_untransformed)):
            all = np.delete(all, np.where(np.isnan(all[i]))[0], axis=0)
            obs = np.delete(obs, np.where(np.isnan(obs[i]))[0], axis=0)

        # remove observations in which the CF is missing or outside of (0,1), or where sample size is missing
        obs = np.delete(obs, np.where((np.isnan(obs.cf)) | (obs.cf > 1) | (obs.cf < 0) | (np.isnan(obs.sample_size)))[0], axis=0)

        # make lists of all the countries/regions/ages/years to predict for
        self.country_list = np.unique(all.country)
        self.region_list = np.unique(all.region)
        self.super_region_list = np.unique(all.super_region)
        self.age_list = np.unique(all.age)
        self.year_list = np.unique(all.year)

        # apply a moving average (5 year window) on cause fractions of 0 or 1, or where sample size is less than 100
        age_lookups = {}
        for a in self.age_list:
            age_lookups[a] = np.where(obs.age == a)[0]
        country_lookups = {}
        country_age_lookups = {}
        for c in self.country_list:
            country_lookups[c] = np.where(obs.country == c)[0]
            for a in self.age_list:
                country_age_lookups[c+'_'+str(a)] = np.intersect1d(country_lookups[c], age_lookups[a])
        year_window_lookups = {}
        for y in range(self.year_range[0],self.year_range[1]+1):
            year_window_lookups[y] = np.where((obs.year >= y-2.) & (obs.year <= y+2.))[0]
        smooth_me = np.where((obs.cf==0.) | (obs.cf==1.) | (obs.sample_size<100.))[0]
        for i in smooth_me:
            obs.cf[i] = obs.cf[np.intersect1d(country_age_lookups[obs.country[i]+'_'+str(obs.age[i])],year_window_lookups[obs.year[i]])].mean()

        # for cases in which the CF is still 0 or 1 after the moving average, use the smallest/largest non-0/1 CF observed in that region-age
        region_age_lookups = {}
        region_lookups = {}
        for r in self.region_list:
            region_lookups[r] = np.where(obs.region == r)[0]
            for a in self.age_list:
                region_age_lookups[str(r)+'_'+str(a)] = np.intersect1d(region_lookups[r], age_lookups[a])
        validcfs = np.where((obs.cf>0.) & (obs.cf<1.))[0]
        for i in np.where(obs.cf==0.)[0]:
            candidates = np.intersect1d(region_age_lookups[str(obs.region[i])+'_'+str(obs.age[i])], validcfs)
            if candidates.shape[0] == 0:
                obs.cf[i] = 0.
            else:
                obs.cf[i] = obs.cf[candidates].min()
        for i in np.where(obs.cf==1.)[0]:
            candidates = np.intersect1d(region_age_lookups[str(obs.region[i])+'_'+str(obs.age[i])], validcfs)
            if candidates.shape[0] == 0:
                obs.cf[i] = 1.
            else:
                obs.cf[i] = obs.cf[candidates].max()

        # finally, any CF that is still 0 or 1 after the above corrections should simply be dropped
        obs = np.delete(obs, np.where((obs.cf == 0.) | (obs.cf == 1.))[0], axis=0)
        
        # make covariate matrices (including transformations and normalization)
        obs_vectors = [obs.country, obs.region, obs.super_region, obs.year, obs.age, obs.cf, obs.sample_size, obs.envelope, obs.pop, np.ones(obs.shape[0])]
        obs_names = ['country', 'region', 'super_region', 'year', 'age', 'cf', 'sample_size', 'envelope', 'pop', 'x0']
        all_vectors = [all.country, all.region, all.super_region, all.year, all.age, all.envelope, all.pop, np.ones(all.shape[0])]
        all_names = ['country', 'region', 'super_region', 'year', 'age', 'envelope', 'pop', 'x0']
        self.covariate_dict = {'x0': 'constant'}
        for i in range(len(self.covariate_list)):
            a = all[self.covariates_untransformed[i]]
            o = obs[self.covariates_untransformed[i]]
            if self.covariate_transformations[i] == 'ln':
                a = np.log(a)
                o = np.log(o)
            elif self.covariate_transformations[i] == 'ln+sq':
                a = (np.log(a))**2
                o = (np.log(o))**2
            elif self.covariate_transformations[i] == 'sq':
                a = a**2
                o = o**2
            if self.normalize == True:
                a = ((a-np.mean(a))/np.std(a))
                o = ((o-np.mean(o))/np.std(o))
            all_vectors.append(a)
            all_names.append('x' + str(i+1))
            obs_vectors.append(o)
            obs_names.append('x' + str(i+1))
            self.covariate_dict['x' + str(i+1)] = self.covariate_list[i]

        # create age dummies if specified
        if self.age_dummies == True:
            pre_ref = 1
            for i,j in enumerate(self.age_list):
                if j == self.age_ref:
                    pre_ref = 0
                elif pre_ref == 1:
                    all_vectors.append(np.array(all.age==j).astype(np.float))
                    all_names.append('x' + str(len(self.covariate_list)+i+1))
                    obs_vectors.append(np.array(obs.age==j).astype(np.float))
                    obs_names.append('x' + str(len(self.covariate_list)+i+1))
                    self.covariate_dict['x' + str(len(self.covariate_list)+i+1)] = 'Age ' + str(j)
                else:
                    all_vectors.append(np.array(all.age==j).astype(np.float))
                    all_names.append('x' + str(len(self.covariate_list)+i))
                    obs_vectors.append(np.array(obs.age==j).astype(np.float))
                    obs_names.append('x' + str(len(self.covariate_list)+i))
                    self.covariate_dict['x' + str(len(self.covariate_list)+i)] = 'Age ' + str(j)
        
        # return the prediction and observation matrices
        self.prediction_matrix = np.core.records.fromarrays(all_vectors, names=all_names)
        self.observation_matrix = np.core.records.fromarrays(obs_vectors, names=obs_names)
            
        # prep all the in-sample data
        self.data_rows = self.observation_matrix.shape[0]
        print 'Data Rows:', self.data_rows
        self.training_split()


    def training_split(self, holdout_unit='none', holdout_prop=.2):
        ''' Splits the data up into test and train subsets '''
        if holdout_prop > .99 or holdout_prop < .01:
            raise ValueError('The holdout proportion must be between .1 and .99.')
        if holdout_unit == 'none':
            self.training_data = self.observation_matrix
            self.test_data = self.observation_matrix
            print 'Fitting model to all data'
        elif holdout_unit == 'datapoint':
            data_flagged = recfunctions.append_fields(self.observation_matrix, 'holdout', np.random.binomial(1, holdout_prop, (self.data_rows,1)))
            self.training_data = np.delete(data_flagged, np.where(data_flagged.holdout==1)[0], axis=0)
            self.test_data = np.delete(data_flagged, np.where(data_flagged.holdout==0)[0], axis=0)
            print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of datapoints'
        elif holdout_unit == 'country-year':
            country_years = [self.observation_matrix.country[i] + '_' + str(self.observation_matrix.year[i]) for i in self.data_rows]
            data_flagged = recfunctions.append_fields(self.observation_matrix, 'holdout', np.zeros((self.data_rows,1)))
            for i in np.unique(country_years):
                data_flagged[np.where(country_years==i)[0]] = np.random.binomial(1, holdout_prop)
            self.training_data = np.delete(data_flagged, np.where(data_flagged.holdout==1)[0], axis=0)
            self.test_data = np.delete(data_flagged, np.where(data_flagged.holdout==0)[0], axis=0)
            print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of country-years'
        elif holdout_unit == 'country':
            data_flagged = recfunctions.append_fields(self.observation_matrix, 'holdout', np.zeros((self.data_rows,1)))
            for i in self.country_list:
                data_flagged.holdout[np.where(country==i)[0]] = np.random.binomial(1, holdout_prop)
            self.training_data = np.delete(data_flagged, np.where(data_flagged.holdout==1)[0], axis=0)
            self.test_data = np.delete(data_flagged, np.where(data_flagged.holdout==0)[0], axis=0)
            print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of countries'
        else:
            raise ValueError("The holdout unit must be either 'datapoint', 'country-year', or 'country'.")
        


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

            Y_c,t,a     ~ observed deaths due to a cause in a country/year/age/sex

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
        beta = mc.Laplace('beta', mu=0.0, tau=1.0, value=np.linalg.lstsq(X.T, np.log(self.training_data.cf))[0])
        # prior on alpha (overdispersion parameter)
        # implemented as alpha = 10^rho; alpha=1 high overdispersion, alpha>10^10=poisson
        rho = mc.Truncnorm('rho', mu=8., tau=.1, a=0., b=11., value=8.)
        # priors on matern amplitudes
        sigma_s = mc.Exponential('sigma_s', beta=2.0, value=2.0)
        sigma_r = mc.Exponential('sigma_r', beta=1.5, value=1.5)
        sigma_c = mc.Exponential('sigma_c', beta=1.0, value=1.0)
        # priors on matern scales
        tau_s = mc.Truncnorm('tau_s', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)
        tau_r = mc.Truncnorm('tau_r', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)
        tau_c = mc.Truncnorm('tau_c', mu=15.0, tau=5.0**-2.0, a=5.0, b=np.Inf, value=15.0)

        # find indices for each subset
        super_regions = self.super_region_list
        s_index = [np.where(self.training_data.super_region==s) for s in super_regions]
        s_list = range(len(super_regions))
        self.super_region_lookup = {}
        for s in s_list:
            self.super_region_lookup[super_regions[s]] = s
        regions = self.region_list
        r_index = [np.where(self.training_data.region==r) for r in regions]
        r_list = range(len(regions))
        self.region_lookup = {}
        for r in r_list:
            self.region_lookup[regions[r]] = r
        countries = self.country_list
        c_index = [np.where(self.training_data.country==c) for c in countries]
        c_list = range(len(countries))
        self.country_lookup = {}
        for c in c_list:
            self.country_lookup[countries[c]] = c
        years = self.year_list
        t_index = dict([(t, i) for i, t in enumerate(years)])
        ages = self.age_list
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
            ''' fixed_effect_c,t,a = beta * X_c,t,a '''
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
        pi_s_samples = [mc.MvNormalCov('pi_s_%s'%s, np.zeros(sample_points.shape[0]), C_s, value=np.zeros(sample_points.shape[0])) for s in s_list]
        pi_r_samples = [mc.MvNormalCov('pi_r_%s'%r, np.zeros(sample_points.shape[0]), C_r, value=np.zeros(sample_points.shape[0])) for r in r_list]
        pi_c_samples = [mc.MvNormalCov('pi_c_%s'%c, np.zeros(sample_points.shape[0]), C_c, value=np.zeros(sample_points.shape[0])) for c in c_list]

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
        @mc.deterministic
        def alpha(rho=rho):
            return 10.**rho
        @mc.observed
        def data_likelihood(value=self.training_data.cf * self.training_data.sample_size, mu=param_pred, alpha=alpha):
            if mu.min() <= 0.:
                tmp = mu.copy()
                tmp[np.where(tmp <= 0.)] = tmp[np.where(tmp > 0.)].min()
                mu = tmp
            if alpha >= 10**10:
                return mc.poisson_like(value, mu)
            else:
                return mc.negative_binomial_like(value, mu, alpha)

        # create a pickle backend to store the model
        #import time as tm
        #dbname = '/home/j/Project/Causes of Death/CoDMod/tmp files/codmod_' + self.cause + '_' + tm.strftime('%b%d_%I%M%p')
        #db = mc.database.pickle.Database(dbname=dbname, dbmode='w')

        # MCMC step methods
        #self.mod_mc = mc.MCMC(vars(), db=db)
        self.mod_mc = mc.MCMC(vars(), db='ram')
        '''
        stochastics = [beta, sigma_s, sigma_r, sigma_c, tau_s, tau_r, tau_c]
        for s in self.mod_mc.pi_s_samples:
            stochastics.append(s)
        for r in self.mod_mc.pi_r_samples:
            stochastics.append(r)
        for c in self.mod_mc.pi_c_samples:
            stochastics.append(c)
        #self.mod_mc.use_step_method(gs.HMCStep, stochastics)
        '''
        self.mod_mc.use_step_method(mc.AdaptiveMetropolis, [self.mod_mc.beta, self.mod_mc.rho, self.mod_mc.sigma_s, self.mod_mc.sigma_r, self.mod_mc.sigma_c, self.mod_mc.tau_s, self.mod_mc.tau_r, self.mod_mc.tau_c])
        
        # use covariance matrix to seed adaptive metropolis steps
        for s in s_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_s_samples[s], cov=np.array(C_s.value*.01))
        for r in r_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01))
        for c in c_list:
            self.mod_mc.use_step_method(mc.AdaptiveMetropolis, self.mod_mc.pi_c_samples[c], cov=np.array(C_c.value*.01))

        # find good initial conditions with MAP approximation
        for var_list in [[self.mod_mc.data_likelihood, self.mod_mc.beta, self.mod_mc.rho]] + \
            [[self.mod_mc.data_likelihood, s] for s in self.mod_mc.pi_s_samples] + \
            [[self.mod_mc.data_likelihood, r] for r in self.mod_mc.pi_r_samples] + \
            [[self.mod_mc.data_likelihood, c] for c in self.mod_mc.pi_c_samples] + \
            [[self.mod_mc.data_likelihood, self.mod_mc.beta, self.mod_mc.rho]]:
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

