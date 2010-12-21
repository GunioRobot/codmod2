'''
# Equivalent R code:
library(mlmRev)
data(Chem97)
m <- lmer(score ~ age + (1|lea:school), Chem97)
ranef(m)
'''

# import data
import numpy as np
import pymc as mc
from pylab import csv2rec
Chem97 = csv2rec('/home/j/Project/Causes of Death/Sandbox/Chem97.csv')

# extract the necessary data
Y_obs   = Chem97.score
X       = Chem97.age.astype(float)
dist    = Chem97.lea
school  = Chem97.school
student = Chem97.column0

# setup the model
''' 
	Y_dsi = beta*X_dsi + pi_ds + e_dsi
		
		pi_ds ~ N(pi_d, sigma_s)
		pi_d  ~ N(0, sigma_d)
	
		e_dsi ~ N(0, sigma_e)
	
	where: 	i = student
			s = school
			d = district (aka local education authority)
'''

# priors
beta = mc.Normal('beta', mu=0., tau=1., value=0.)
sigma_s = mc.Exponential('sigma_s', beta=1., value=1.)
sigma_d = mc.Exponential('sigma_d', beta=1., value=1.)
sigma_e = mc.Exponential('sigma_e', beta=1., value=1.)

# create random effects for each district
pi_d = [mc.Normal('pi_d%s'%d, mu=0., tau=sigma_d, value=0.) for d in np.unique(dist)]

# find the school random effects within each district
pi_ds = []
for s in np.unique(school) :
	# district for this school
	d = np.unique(dist[school==s])
	# create random effect for this school
	pi_ds.append(mc.Normal('pi_s%s'%s, mu=pi_d[np.where([np.unique(dist)==d])[0][0]], tau=sigma_d, value=0.))

# predict the score for each student based on beta*X and random effects
param_preds = []
for i in student : 
	# school for this student
	s = school[student==i]
	@mc.deterministic
	def param_pred(mu=mu, s=s, i=i):
		# find school random effect for student i
		pi_i = pi_ds[np.where([np.unique(school)==s])[0][0]]
		# find fixed effect prediction for student i
		mu_i = beta*X[np.where(student==i)]
		# return the sum of fixed and random effects
		param_pred = np.zeros_like(Y_obs)
		param_pred[np.where(student==i)] = pi_i + mu_i
		return param_pred
	param_preds.append(param_pred)
@mc.deterministic
def predicted(param_preds=param_preds):
	return pl.sum(country_age_param_pred, axis=0)

# observe the data
@mc.observed
def obs(value=Y_obs) :
	return value

# run the model
mod_mc = mc.MCMC(vars())
	


# level 2 model
'''
'''
