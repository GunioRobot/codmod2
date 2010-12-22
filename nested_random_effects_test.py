'''
# Equivalent R code:
library(mlmRev)
data(Chem97)
Chem97 <- Chem97[as.numeric(Chem97$lea)<=20,]
m <- lmer(score ~ age + (1|lea:school), Chem97)
ranef(m)
'''

# import data
import numpy as np
import pymc as mc
from pylab import csv2rec
Chem97 = csv2rec('/home/j/Project/Causes of Death/Sandbox/Chem97.csv')
Chem97 = Chem97[Chem97.lea<=20,]

# extract the necessary data
Y_obs   = Chem97.score.astype(float)
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

# find fixed effect prediction
@mc.deterministic
def mu(X=X, beta=beta) :
	return np.dot(beta, X)

# create random effects for each district
pi_d = [mc.Normal('pi_d%s'%d, mu=0., tau=sigma_d, value=0.) for d in np.unique(dist)]

# find the school random effects within each district
pi_ds = []
for s in np.unique(school) :
	# district for this school
	d = np.unique(dist[school==s])
	# create random effect for this school
	pi_ds.append(mc.Normal('pi_s%s'%s, mu=pi_d[np.where([np.unique(dist)==d])[0][0]], tau=sigma_d, value=0.))

# predict the score for each student based on fixed and random effects
param_preds = []
for s in np.unique(school) :
	print(s)
	# find indices for this school
	s_i = np.where(school==s)[0]
	s_s = np.where(np.unique(school)==s)[0]
	# add up fixed/random effects
	@mc.deterministic(name='param_pred_%s'%s)
	def param_pred_s(s=s, mu=mu, pi_ds=pi_ds) :
		param_pred_s = np.zeros_like(X)
		param_pred_s[s_i] = mu[s_i] + pi_ds[s_s]
		return param_pred_s
	param_preds.append(param_pred_s)
@mc.deterministic
def predicted(param_preds=param_preds) :
	return np.sum(param_preds, axis=0)


# observe the data
@mc.observed
def obs(value=Y_obs, sigma_e=sigma_e, predicted=predicted) :
	return mc.normal_like(value, predicted, 1/sigma_e**2)


# fit the model via MAP
mod_mc = mc.MCMC(vars())
mc.MAP(mod_mc).fit(verbose=1, iterlim=100000)

