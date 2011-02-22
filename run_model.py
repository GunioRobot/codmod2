# load the codmod program
import codmod as cm
reload(cm)

# setup the model
m = cm.codmod('Ab10','female')
m.set_covariates(covariate_list=['year','education_yrs_pc','ln(TFR)','neonatal_deaths_per1000','ln(LDI_pc)','HIV_prevalence_pct'], age_dummies=True, age_ref=30, normalize=True)
m.set_window(age_range=[15,45], year_range=[1980,2010])
m.set_pi_samples(age_samples=[15,25,35,45], year_samples=[1980,1990,2000,2010])

m.load()

m.initialize_model()

m.sample()


