if c(os) == "Windows" global j "J:"
else global j "/home/j"

insheet using "$j/Project/Causes of Death/CoDMod/tmp/mapplusnorm_predict_predictions_Ab10_female.csv", comma clear
rename country iso3

levelsof iso3, l(isos) c
foreach i of local isos {
	cap merge 1:1 iso3 year age using "$j/Project/Causes of Death/CoDMod/Predictive Validity/Ab10/Maternal_Rerun/Results/agespecific_1_F_`i'.dta", nogen update
}

levelsof super_region, l(srs) c
foreach s of local srs {
	merge m:m iso3 year age using "$j/Project/Causes of Death/CoDMod/Predictive Validity/Ab10/Maternal_Rerun/Results/all/step1_1_F_`s'_all.dta", nogen update
}

gen mean_new = real(mean_deaths) / real(pop) * 100000
gen upper_new = real(upper_deaths) / real(pop) * 100000
gen lower_new = real(lower_deaths) / real(pop) * 100000

gen mean_old = exp(mean_all_sc10_am1_vm1)
gen upper_old = exp(upper_all_sc10_am1_vm1)
gen lower_old = exp(lower_all_sc10_am1_vm1)

gen actual = exp(ln_rate)

do "$j/Usable/Tools/ADO/pdfmaker.do"
pdfstart using "$j/Project/Causes of Death/CoDMod/tmp/comparison_plots.pdf"
foreach i of local isos {
	preserve
	keep if iso3 == "`i'"
	scatter actual year, by(age, legend(off) style(compact) title(`i') yrescale) mcolor(black) msize(small) || line mean_new lower_new upper_new year, lcolor(blue blue blue) lpattern(solid dash dash) || line mean_old lower_old upper_old year, lcolor(red red red) lpattern(solid dash dash)
	pdfappend
	restore
}
pdffinish





