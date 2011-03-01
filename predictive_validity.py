'''
Author:     Kyle Foreman
Date:       February 28, 2011
Purpose:    Run codmod2 on maternal for several different predictive validity tests
'''

modname = 'maternal'
import os
for t in ['country1', 'country2', 'country3', 'country-year1', 'country-year2', 'country-year3', 'datapoint1', 'datapoint2', 'datapoint3','none1']:
    if t[:-1] == 'none':
        tname = 'predict'
    else:
        tname = t

    # write a file to run each test
    f = open('/home/j/Project/Causes of Death/CoDMod/tmp/' + modname + '_' + tname + '.py', 'w')
    f.write("import os" + "\n")
    f.write("os.system('hostname')" + "\n")
    f.write("os.chdir('/home/j/Project/Causes of Death/CoDMod/codmod2')" + "\n")
    f.write("import codmod as cm" + "\n")
    f.write("m = cm.codmod('Ab10','female','" + modname + "_" + tname + "')" + "\n")
    f.write("m.set_covariates(covariate_list=['year','education_yrs_pc','ln(TFR)','neonatal_deaths_per1000','ln(LDI_pc)','HIV_prevalence_pct'], age_dummies=True, age_ref=30, normalize=True)" + "\n")
    f.write("m.set_window(age_range=[15,45], year_range=[1980,2010])" + "\n")
    f.write("m.set_pi_samples(age_samples=[15,25,35,45], year_samples=[1980,1990,2000,2010])" + "\n")
    f.write("m.load(use_cache=True)" + "\n")
    f.write("m.training_split(holdout_unit='" + t[:-1] + "', holdout_prop=.2)" + "\n")
    f.write("m.initialize_model(find_start_vals=True)" + "\n")
    f.write("m.sample(iter=4000, burn=0, thin=5)" + "\n")
    if tname == 'predict':
        f.write("m.predict_test(save_csv=True)" + "\n")
        f.write("m.mcmc_diagnostics()" + "\n")
    else:
        f.write("m.predict_test(save_csv=True)" + "\n")
        f.write("m.measure_fit()" + "\n")
    f.close()

    # write a shell script to run this test on the cluster
    f = open('/home/j/Project/Causes of Death/CoDMod/tmp/run_' + modname + '_' + tname + '.sh', 'w')
    f.write("#!/bin/sh" + "\n")
    f.write("#$ -S /bin/sh" + "\n")
    f.write("/usr/local/epd_py25-4.3.0/bin/ipython '/home/j/Project/Causes of Death/CoDMod/tmp/" + modname + "_" + tname + ".py'" + "\n")
    f.close()
    
    # submit the jobs to the cluster
    os.system("qsub -o '/home/j/Project/Causes of Death/CoDMod/tmp/' -e '/home/j/Project/Causes of Death/CoDMod/tmp/' -pe multi_slot 4 -l mem_free=12G '/home/j/Project/Causes of Death/CoDMod/tmp/run_" + modname + "_" + tname + ".sh'")
