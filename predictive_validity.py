'''
Author:     Kyle Foreman
Date:       February 26, 2011
Purpose:    Run codmod2 on maternal for several different predictive validity tests
'''

import os
for t in ['country1', 'country2', 'country3', 'country-year1', 'country-year2', 'country-year3', 'datapoint1', 'datapoint2', 'datapoint3','none1']:
    if t[:-1] == 'none':
        tname = 'predict'
    else:
        tname = t
    
    # write a file to run each test
    f = open('/home/j/Project/Causes of Death/CoDMod/tmp/maternal_' + tname + '.py', 'w')
    f.write("import os" + "\n")
    f.write("os.system('hostname')" + "\n")
    f.write("os.chdir('/home/j/Project/Causes of Death/CoDMod/codmod2')" + "\n")
    f.write("import codmod as cm" + "\n")
    f.write("m = cm.codmod('Ab10','female','maternal_" + tname + "')" + "\n")
    f.write("m.set_covariates(covariate_list=['year','education_yrs_pc','ln(TFR)','neonatal_deaths_per1000','ln(LDI_pc)','HIV_prevalence_pct'], age_dummies=True, age_ref=30, normalize=True)" + "\n")
    f.write("m.set_window(age_range=[15,45], year_range=[1980,2010])" + "\n")
    f.write("m.set_pi_samples(age_samples=[15,25,35,45], year_samples=[1980,1990,2000,2010])" + "\n")
    f.write("m.load(use_cache=True)" + "\n")
    f.write("m.training_split(holdout_unit='" + t[:-1] + "', holdout_prop=.2)" + "\n")
    f.write("m.initialize_model(find_start_vals=True)" + "\n")
    f.write("m.sample(iter=2000, burn=0, thin=1)" + "\n")
    if tname == 'predict':
        f.write("m.predict_test(save_csv=True)" + "\n")
        f.write("m.mcmc_diagnostics()" + "\n")
    else:
        f.write("m.predict_test(save_csv=False)" + "\n")
        f.write("m.measure_fit()" + "\n")
    f.close()

    # write a shell script to run this test on the cluster
    f = open('/home/j/Project/Causes of Death/CoDMod/tmp/run_maternal_' + tname + '.sh', 'w')
    f.write("#!/bin/sh" + "\n")
    f.write("#$ -S /bin/sh" + "\n")
    f.write("/usr/local/epd_py25-4.3.0/bin/ipython '/home/j/Project/Causes of Death/CoDMod/tmp/maternal_" + tname + ".py'" + "\n")
    f.close()
    
    # submit the jobs to the cluster
    os.system("qsub -q all.q@ihme001,all.q@ihme001,all.q@ihme002,all.q@ihme003,all.q@ihme004,all.q@ihme005,all.q@ihme006,all.q@ihme007,all.q@ihme008,all.q@ihme009,all.q@ihme010,all.q@ihme011,all.q@ihme012,all.q@ihme013,all.q@ihme015,all.q@ihme016,all.q@ihme017,all.q@ihme018,all.q@ihme019,all.q@ihme020,all.q@ihme021,all.q@ihme022,all.q@ihme023,all.q@ihme024,all.q@ihme025,all.q@ihme026,all.q@ihme027,all.q@ihme028,all.q@ihme029,all.q@ihme030,all.q@ihme031,all.q@ihme032,all.q@ihme033,all.q@ihme034,all.q@ihme035,all.q@ihme036,all.q@ihme037 -l mem_free=20G '/home/j/Project/Causes of Death/CoDMod/tmp/run_maternal_" + tname + ".sh'")
