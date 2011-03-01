# load the codmod program
import os
os.chdir('/home/j/Project/Causes of Death/CoDMod/codmod2/')
import codmod as cm
reload(cm)

# setup multithreading
#import mkl
#mkl.set_num_threads(1)

# setup the model
m = cm.codmod('Ab10','female','maternal_test')
m.set_covariates(covariate_list=['year','education_yrs_pc','ln(TFR)','neonatal_deaths_per1000','ln(LDI_pc)','HIV_prevalence_pct'], age_dummies=True, age_ref=30, normalize=True)
m.set_window(age_range=[15,45], year_range=[1980,2010])
m.set_pi_samples(age_samples=[15,25,35,45], year_samples=[1980,1990,2000,2010])

# load in the data
m.load(use_cache=True)
m.training_split(holdout_unit='country')

# build the model and use MAP to find starting values
m.initialize_model(find_start_vals=True)

# use MCMC to find posterior
m.sample(iter=2, burn=0, thin=1)

# make predictions
m.predict_test(save_csv=False)

# find fits
m.measure_fit()

# make MCMC diagnostic plots
#m.mcmc_diagnostics()

