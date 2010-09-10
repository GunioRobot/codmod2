""" Module to test new code, and also to automatically test that working things stay working"""

# matplotlib backend setup, so that graphics can run on cluster (which doesn't have X11)
import matplotlib
matplotlib.use("AGG") 

import pylab as pl
import time


data_gen_models = 'fe re gp_re nre ngp_re ngp_re_a smooth_gp_re_a smooth_ngp_re_a_full'
def regenerate_data(data_model, pct=80., std=1.):
    """ Regenerate test data using specified data generation function
    Allowed models: %s """ % data_gen_models
    if data_model not in data_gen_models.split(' '):
        raise TypeError, 'Unrecognized data model "%s"; must be one of %s' % (data_model, data_gen_models)
    
    import data
    reload(data)
    eval('data.generate_%s()' % data_model)
    data.add_sampling_error(std=std)
    data.knockout_uniformly_at_random(pct=pct)

data_run_models = 'fe re nested_re gp_re gp_re_a gp_re2 nested_gp_re nested_gp_re_a nested_gp_re2'
def evaluate_model(mod, comment='', data_fname='missing_noisy_data.csv', truth_fname='data.csv'):
    """ Run specified model on existing data (data.csv / missing_noisy_data.csv) and save results in dev_log.csv
    Existing models: %s """ % data_run_models
    if mod not in data_run_models.split(' '):
        raise TypeError, 'Unrecognized model "%s"; must be one of %s' % (mod, data_run_models)

    import model
    reload(model)

    print 'loading data'
    data = pl.csv2rec(data_fname)
    truth = pl.csv2rec(truth_fname, skiprows=1)  # skiprows hack, for this old version of csv2rec
    
    t0 = time.time()
    print 'generating model'
    mod_mc = eval('model.%s(data)' % mod)

    print 'fitting model with mcmc'
    #mod_mc.sample(10000, 5000, 50, verbose=1)
    mod_mc.sample(100, verbose=1)
    t1 = time.time()

    print 'summarizing results'

    import graphics
    reload(graphics)
    pl.figure(figsize=(22, 17), dpi=300)
    pl.clf()
    graphics.plot_all_predictions_over_time(data, mod_mc.predicted, more_data=truth)

    stats = mod_mc.predicted.stats()
    i_out = [i for i in range(len(data)) if pl.isnan(data.y[i])]
    rmse_abs_out = pl.rms_flat(truth.y[i_out] - stats['mean'][i_out])
    rmse_rel_out = 100*pl.rms_flat(1. - stats['mean'][i_out]/truth.y[i_out])

    i_in = [i for i in range(len(data)) if not pl.isnan(data.y[i])]
    rmse_abs_in = pl.rms_flat(truth.y[i_in] - stats['mean'][i_in])
    rmse_rel_in = 100*pl.rms_flat(1. - stats['mean'][i_in]/truth.y[i_in])

    stats = mod_mc.param_predicted.stats()
    coverage = 100*pl.sum((truth.y[i_out] >= stats['95% HPD interval'][i_out, 0]) & (truth.y[i_out] <= stats['95% HPD interval'][i_out, 1])) / float(len(i_out))

    import md5
    data_hash = md5.md5(data).hexdigest()
    data_comment = open('data.csv').readline()
    
    results = [mod, t1-t0, rmse_abs_out, rmse_rel_out, rmse_abs_in, rmse_rel_in, coverage,
               len(data), len(pl.unique(data.region)), len(pl.unique(data.country)), len(pl.unique(data.year)), len(pl.unique(data.age)), data_hash,
               t0, data_comment, comment]
    print '%s: time: %.0fs out-of-samp rmse abs=%.1f rel=%.0f in-samp rmse abs=%.1f rel=%.0f coverage=%.0f\ndata: %d rows; %d regions, %d countries %d years %d ages [data hash: %s]\n(run conducted at %f) %s - %s' % tuple(results)

    pl.savefig('/home/j/Project/Models/space-time-smoothing/images/%s.png' % t0)  # FIXME: don't hardcode path for saving images

    import csv
    f = open('dev_log.csv', 'a')
    f_csv = csv.writer(f)
    f_csv.writerow(results)
    f.close()

    return mod_mc

def run_all_models(data, testing=False):
    """ Run models for testing and comparison
    """
    mc_dict = {}
    for mod in [fe, re, nested_re, gp_re, gp_re_a, gp_re2, nested_gp_re, nested_gp_re_a, nested_gp_re2]:
        print "setting up model (%s)" % mod
        mod_mc = mod(data)

        print "sampling with MCMC"
        if testing == True:
            mod_mc.sample(iter=2)
        else:
            mod_mc.sample(iter=20000, burn=10000, thin=100, verbose=1)

        print "saving results"
        mc_dict[mod] = mod_mc

    return mc_dict


def run_all_data_generators():
    """ Test data generating module
    """

    generate_fe('test_data.csv')  # replace data.csv with file based on fixed-effects model
    generate_re('test_data.csv')
    generate_nre('test_data.csv')
    generate_gp_re('test_data.csv')
    generate_ngp_re('test_data.csv')
    generate_ngp_re_a('test_data.csv')
    generate_smooth_gp_re_a('test_data.csv')
    generate_smooth_ngp_re_a_full('test_data.csv')

    # add normally distributed noise (with standard deviation 1.0) to test_data.y
    add_sampling_error('test_data.csv',
                       'noisy_test_data.csv',
                       std=1.)

    # replace new_data.csv by knocking out data uar from data.csv
    knockout_uniformly_at_random('noisy_test_data.csv',
                                 'missing_noisy_test_data.csv',
                                 pct=25.)


def run_all_graphics(data, mc_dict):
    """ Test plots for all mcmc results in the mc_dict
    """

    for mod, mod_mc in mc_dict.items():
        print 'testing plots of mod %s' % mod
        plot_prediction_over_time('USA', data, mod_mc.predicted)  # test single plot of model predictions
        plot_all_predictions_over_time(data, mod_mc.predicted)  # test single plot of model predictions

def test():
    """ Test that the models all run, data generation works, and graphics functions work
    """
    print run_all_data_generators.__doc__
    run_all_data_generators()

    print run_all_models.__doc__
    data = pl.csv2rec('missing_noisy_test_data.csv')
    mc_dict = run_all_models(data, testing=True) # test fitting models

    print run_all_graphics.__doc__
    run_all_graphics(data, mc_dict)

if __name__ == '__main__':
    import pylab as pl
    import data

    age_range = pl.arange(0, 81, pl.randint(1, 5)*5)
    time_range = pl.arange(1980, 2005, pl.randint(1, 6))
    regions = pl.randint(1, 22)

    std=pl.rand()*5.
    pct=pl.randint(0,100)

    print age_range, time_range, regions, std, pct
    
    time.sleep(pl.rand()*5.)
    t0 = time.time()
    data.generate_smooth_gp_re_a('test_data/%s.csv'%t0)
    data.add_sampling_error('test_data/%s.csv'%t0, 'test_data/noisy_%s.csv'%t0, std=std)
    data.knockout_uniformly_at_random('test_data/noisy_%s.csv'%t0, 'test_data/missing_noisy_%s.csv'%t0, pct=pct)

    evaluate_model('gp_re_a', 'replication to test speed and accuracy (std=%.3f, pct=%d)' % (std, pct), 'test_data/missing_noisy_%s.csv'%t0, 'test_data/%s.csv'%t0)