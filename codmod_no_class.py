'''
Author:	    Kyle Foreman
Date:	    28 February 2011
Purpose:    Fit cause of death models over space, time, and age
'''

import matplotlib
matplotlib.use("AGG") 
import pymc as mc
import numpy as np
import pylab as pl
import MySQLdb
from scipy import interpolate
import numpy.lib.recfunctions as recfunctions
import time as tm
from multiprocessing import Pool

# setup parameters
cause = 'Ab10'
sex = 'female'
age_range = [15,45]
year_range = [1980,2010]
age_samples = [15,25,35,45]
year_samples = [1980,1990,2000,2010]
covariate_list = ['year','education_yrs_pc','ln(TFR)','neonatal_deaths_per1000','ln(LDI_pc)','HIV_prevalence_pct']
age_dummies = True
age_ref = 30
normalize = True
holdout_unit = 'none'
holdout_prop = .2
num_threads = 4

# connect to mysql
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
mysql_opts = open('./mysql.cnf')
for l in mysql_opts:
    exec l
mysql = MySQLdb.connect(host=host, db=db, user=user, passwd=passwd)
cursor = mysql.cursor()

# parse covariates
if type(covariate_list) == str:
    covariate_list = [covariate_list]
covariates_untransformed = []
covariate_transformations = []
for c in covariate_list:
    if c[0:3] == 'ln(' and c[-2:] != '^2':
        covariates_untransformed.append(c[3:len(c)-1])
        covariate_transformations.append('ln')
    elif c[0:3] == 'ln(' and c[-2:] == '^2':
        covariates_untransformed.append(c[3:len(c)-3])
        covariate_transformations.append('ln+sq')
    elif c[-2:] == '^2':
        covariates_untransformed.append(c[:len(c)-3])
        covariate_transformations.append('sq')
    else:
        covariates_untransformed.append(c)
        covariate_transformations.append('none')

# import data
dir = '/home/j/project/causes of death/codmod/tmp/'
prediction_matrix = pl.csv2rec(dir + 'prediction_matrix_' + cause + '_' + sex + '.csv')
observation_matrix = pl.csv2rec(dir + 'observation_matrix_' + cause + '_' + sex + '.csv')
data_rows = observation_matrix.shape[0]
country_list = np.unique(prediction_matrix.country)
region_list = np.unique(prediction_matrix.region)
super_region_list = np.unique(prediction_matrix.super_region)
age_list = np.unique(prediction_matrix.age)
year_list = np.unique(prediction_matrix.year)


# load age weights
age_weights = mysql_to_recarray(cursor, 'SELECT age,weight FROM age_weights;')
age_weights = recfunctions.append_fields(age_weights, 'keep', np.zeros(age_weights.shape[0])).view(np.recarray)
for a in age_list:
    age_weights.keep[np.where(age_weights.age==a)[0]] = 1
age_weights = np.delete(age_weights, np.where(age_weights.keep==0)[0], axis=0)
age_weights.weight = age_weights.weight/age_weights.weight.sum()

# perform training/test splits
if holdout_prop > .99 or holdout_prop < .01:
    raise ValueError('The holdout proportion must be between .1 and .99.')
if holdout_unit == 'none':
    training_data = observation_matrix
    test_data = prediction_matrix
    training_type = 'make predictions'
    print 'Fitting model to all data'
elif holdout_unit == 'datapoint':
    holdouts = np.random.binomial(1, holdout_prop, data_rows)
    training_data = np.delete(observation_matrix, np.where(holdouts==1)[0], axis=0)
    test_data = np.delete(observation_matrix, np.where(holdouts==0)[0], axis=0)
    training_type = 'datapoint'
    print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of datapoints'
elif holdout_unit == 'country-year':
    country_years = [observation_matrix.country[i] + '_' + str(observation_matrix.year[i]) for i in range(data_rows)]
    data_flagged = recfunctions.append_fields(observation_matrix, 'holdout', np.zeros(data_rows)).view(np.recarray)
    for i in np.unique(country_years):
        data_flagged.holdout[np.where(data_flagged.country + '_' + data_flagged.year.astype('|S4')==i)[0]] = np.random.binomial(1, holdout_prop)
    training_data = np.delete(data_flagged, np.where(data_flagged.holdout==1)[0], axis=0)
    test_data = np.delete(data_flagged, np.where(data_flagged.holdout==0)[0], axis=0)
    training_type = 'country-year'
    print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of country-years'
elif holdout_unit == 'country':
    data_flagged = recfunctions.append_fields(observation_matrix, 'holdout', np.zeros(data_rows)).view(np.recarray)
    for i in country_list:
        data_flagged.holdout[np.where(data_flagged.country==i)[0]] = np.random.binomial(1, holdout_prop)
    training_data = np.delete(data_flagged, np.where(data_flagged.holdout==1)[0], axis=0)
    test_data = np.delete(data_flagged, np.where(data_flagged.holdout==0)[0], axis=0)
    training_type = 'country'
    print 'Fitting model to ' + str((1-holdout_prop)*100) + '% of countries'
else:
    raise ValueError("The holdout unit must be either 'datapoint', 'country-year', or 'country'.")

'''
Y_c,t,a ~ NegativeBinomial(mu_c,t,a, alpha)

    where   s: super-region
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
                          Binomial(n = total deaths in country, p = proportion recorded in study)
                
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
k = len([n for n in training_data.dtype.names if n.startswith('x')])
X = np.array([training_data['x%d'%i] for i in range(k)])

# prior on beta (covariate coefficients)
beta = mc.Laplace('beta', mu=0.0, tau=1.0, value=np.linalg.lstsq(X.T, np.log(training_data.cf))[0])
# prior on alpha (overdispersion parameter)
# implemented as alpha = 10^rho; alpha=1 high overdispersion, alpha>10^10=poisson
rho = mc.Normal('rho', mu=8.0, tau=.1, value=8.0)
# priors on matern amplitudes
sigma_s = mc.Exponential('sigma_s', beta=2.0, value=2.0)
sigma_r = mc.Exponential('sigma_r', beta=1.5, value=1.5)
sigma_c = mc.Exponential('sigma_c', beta=1.0, value=1.0)
# priors on matern scales
tau_s = mc.Uniform('tau_s', lower=5.0, upper=50.0, value=15.0)
tau_r = mc.Uniform('tau_r', lower=5.0, upper=50.0, value=15.0)
tau_c = mc.Uniform('tau_c', lower=5.0, upper=50.0, value=15.0)

# find indices for each subset
super_regions = super_region_list
s_index = [np.where(training_data.super_region==s) for s in super_regions]
s_list = range(len(super_regions))
regions = region_list
r_index = [np.where(training_data.region==r) for r in regions]
r_list = range(len(regions))
countries = country_list
c_index = [np.where(training_data.country==c) for c in countries]
c_list = range(len(countries))
years = year_list
t_index = dict([(t, i) for i, t in enumerate(years)])
ages = age_list
a_index = dict([(a, i) for i, a in enumerate(ages)])
t_by_s = [[t_index[training_data.year[j]] for j in s_index[s][0]] for s in s_list]
a_by_s = [[a_index[training_data.age[j]] for j in s_index[s][0]] for s in s_list]
t_by_r = [[t_index[training_data.year[j]] for j in r_index[r][0]] for r in r_list]
a_by_r = [[a_index[training_data.age[j]] for j in r_index[r][0]] for r in r_list]
t_by_c = [[t_index[training_data.year[j]] for j in c_index[c][0]] for c in c_list]
a_by_c = [[a_index[training_data.age[j]] for j in c_index[c][0]] for c in c_list]	

# fixed-effect predictions
@mc.deterministic
def fixed_effect(X=X, beta=beta):
    ''' fixed_effect_c,t,a = beta * X_c,t,a '''
    return np.dot(beta, X)

# find all the points on which to evaluate the random effects grid
sample_points = []
for a in age_samples:
    for t in year_samples:
        sample_points.append([a,t])
sample_points = np.array(sample_points)

# choose the degree for spline fitting (prefer cubic, but for undersampling pick smaller)
kx = 3 if len(age_samples) > 3 else len(age_samples)-1
ky = 3 if len(year_samples) > 3 else len(year_samples)-1

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
def interpolate_grid(pi_samples):
    interpolator = interpolate.bisplrep(x=sample_points[:,0], y=sample_points[:,1], z=pi_samples, xb=ages[0], xe=ages[-1], yb=years[0], ye=years[-1], kx=kx, ky=ky)
    return interpolate.bisplev(x=ages, y=years, tck=interpolator)

def find_pi(pi_samples):
    P = Pool(num_threads)
    interp = P.map_async(interpolate_grid, pi_samples, chunksize=np.ceil(len(pi_samples)/num_threads).astype(np.int))
    pi_list = interp.get()
    P.close()
    P.join()
    return pi_list



@mc.deterministic
def pi_s_list(pi_samples=pi_s_samples):
    flattened = []
    for p in range(len(pi_samples)):
        flattened.append(pi_samples[p].tolist())
    return find_pi(flattened)

@mc.deterministic
def pi_s(pi_list=pi_s_list):
    pi_s = np.zeros(training_data.shape[0])
    for s in s_list:
        pi_s[s_index[s]] = pi_list[s][a_by_s[s],t_by_s[s]]
    return pi_s

@mc.deterministic
def pi_r_list(pi_samples=pi_r_samples):
    flattened = []
    for p in range(len(pi_samples)):
        flattened.append(pi_samples[p].tolist())
    return find_pi(flattened)

@mc.deterministic
def pi_r(pi_list=pi_r_list):
    pi_r = np.zeros(training_data.shape[0])
    for r in r_list:
        pi_r[r_index[r]] = pi_list[r][a_by_r[r],t_by_r[r]]
    return pi_r

@mc.deterministic
def pi_c_list(pi_samples=pi_c_samples):
    flattened = []
    for p in range(len(pi_samples)):
        flattened.append(pi_samples[p].tolist())
    return find_pi(flattened)

@mc.deterministic
def pi_c(pi_list=pi_c_list):
    pi_c = np.zeros(training_data.shape[0])
    for c in c_list:
        pi_c[c_index[c]] = pi_list[c][a_by_c[c],t_by_c[c]]
    return pi_c

# estimation of exposure based on coverage
p = training_data.sample_size / training_data.envelope
E = mc.Binomial('E', n=np.round(training_data.envelope), p=p, value=np.round(training_data.sample_size))

# parameter predictions
@mc.deterministic
def param_pred(fixed_effect=fixed_effect, pi_s=pi_s, pi_r=pi_r, pi_c=pi_c, E=E):
    return np.exp(np.vstack([fixed_effect, np.log(E), pi_s, pi_r, pi_c]).sum(axis=0))

# observe the data
@mc.deterministic
def alpha(rho=rho):
    return 10.**rho
@mc.observed
def data_likelihood(value=np.round(training_data.cf * training_data.sample_size), mu=param_pred, alpha=alpha):
    if alpha >= 10**10:
        return mc.poisson_like(value, mu)
    else:
        if mu.min() <= 0.:
            mu = mu + 10**-10
        return mc.negative_binomial_like(value, mu, alpha)

 
mod_mc = mc.MCMC(vars(), db='ram')

# MCMC step methods
mod_mc.use_step_method(mc.AdaptiveMetropolis, [mod_mc.beta, mod_mc.rho, mod_mc.E, mod_mc.sigma_s, mod_mc.sigma_r, mod_mc.sigma_c, mod_mc.tau_s, mod_mc.tau_r, mod_mc.tau_c], interval=100)
for s in s_list:
    mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_s_samples[s], cov=np.array(C_s.value*.01), interval=100)
for r in r_list:
    mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_r_samples[r], cov=np.array(C_r.value*.01), interval=100)
for c in c_list:
    mod_mc.use_step_method(mc.AdaptiveMetropolis, mod_mc.pi_c_samples[c], cov=np.array(C_c.value*.01), interval=100)

# find good initial conditions with MAP approximation
if find_start_vals==True:
    for var_list in [[mod_mc.data_likelihood, mod_mc.beta, mod_mc.rho]] + \
        [[mod_mc.data_likelihood, s] for s in mod_mc.pi_s_samples] + \
        [[mod_mc.data_likelihood, r] for r in mod_mc.pi_r_samples] + \
        [[mod_mc.data_likelihood, c] for c in mod_mc.pi_c_samples] + \
        [[mod_mc.data_likelihood, mod_mc.beta, mod_mc.rho]]:
        print 'attempting to maximize likelihood of %s' % [v.__name__ for v in var_list]
        mc.MAP(var_list).fit(method='fmin_powell', verbose=1)
        print ''.join(['%s: %s\n' % (v.__name__, v.value) for v in var_list[1:]])



