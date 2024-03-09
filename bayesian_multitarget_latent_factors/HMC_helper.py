import numpy as _np
import scipy.special as _sps
import cmdstanpy as _cmdstanpy
import arviz as _az
import datetime as _datetime
import logging as _logging
import json as _json
import os as _os
import shutil as _shutil
import xarray as _xr


# interactive plotting tools
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual


__all__ = ['change_log_state', 'recompile_all', 'colored_print', 'show_data_block', 'show_dictionary_by_accordion',
           'execute_HMC', 'load_HMC', 'execute_HMC_prior_posterior', 'load_HMC_prior_posterior', 'get_expected_log_likelihood',
           'MAP_and_LaplaceSample', ]

def change_log_state():
    """
    Toggles the logging state of the 'cmdstanpy' logger. If the logger is currently enabled, it will be disabled, and vice versa.

    No parameters.

    Returns:
        None
    """
    logger = _logging.getLogger('cmdstanpy')
    logger.disabled = not logger.disabled


def recompile_all(directory = './stan_codes'):
    """
    Recompiles all Stan models in the specified directory. This involves removing any existing compiled models and recompiling from the source .stan files.

    Args:
        directory (str): The path to the directory containing Stan model files. Defaults to './stan_codes'.

    Returns:
        None
    """    
    for path in [directory + '/' + file for file in _os.listdir(directory) if file[-5:]!='.stan']:
        try:
            _os.remove(path)
        except Exception as e:
            print(e)
            pass

    for path in [directory + '/' + file for file in _os.listdir(directory) if file[-5:]=='.stan']:
        print('Compiling' + path + '\tTime: ' + str( _datetime.datetime.now() ))
        _cmdstanpy.CmdStanModel(stan_file=path, stanc_options={'O1':''})


def colored_print(stringa):
    """
    Prints the given string to the console, coloring any text that follows '//' in red. This function is useful for emphasizing comments in a block of text.

    Args:
        stringa (str): The string to be printed with color formatting applied to comments.

    Returns:
        None
    """
    for row in stringa.split('\n'):
        if len(row.split('//')) == 1:
            print(row)
        else:
            col = row.split('//')
            print(col[0] + "\x1b[31m" + '//' + col[1] + "\x1b[0m")    
    
        
def show_data_block(filename = './stan_codes/mean_estimation_prior.stan'):
    """
    Displays the 'data' block from a specified Stan model file. Comments within the 'data' block are colored red.

    Args:
        filename (str): The path to the Stan model file. Defaults to './stan_codes/mean_estimation_prior.stan'.

    Returns:
        None
    """    
    try:
        file = open(filename)
        colored_print( 'data' + file.read().split('\ndata')[1].split('}')[0] + '}' )
        file.close()
    except Exception as e:
        print(e)
        pass


def show_dictionary_by_accordion(data):
    """
    Self explanitory
    """
    # Create an Accordion widget
    accordion = widgets.Accordion(children=[widgets.Output() for _ in data.keys()])

    # Assign titles and populate the accordion sections
    for i, (title, content) in enumerate(data.items()):
        accordion.set_title(i, title)
        with accordion.children[i]:
            display(content)

    # Display the accordion
    display(accordion)
    


def execute_HMC(data_dic,
                rng_seed,
                stan_file = './stan_codes/mean_estimation_prior.stan',
                output_dir = './out',
                iter_warmup = 1000,
                iter_sampling = 1000,
                inits = None,
                max_treedepth = 10,
                lista_observed = [],
                lista_lklhood = [],
                lista_predictive = [],
                ):
    """
    Executes Hamiltonian Monte Carlo (HMC) sampling for a given Stan model, with specified data, seed, and sampling parameters.

    Args:
        data_dic (dict): The data to be used in the model, in dictionary format.
        rng_seed (int): The seed for the random number generator.
        stan_file (str): The path to the Stan model file.
        output_dir (str): The directory where output files will be saved.
        iter_warmup (int): The number of warmup iterations.
        iter_sampling (int): The number of sampling iterations.
        inits (str or dict): Initialization values for model parameters.
        max_treedepth (int): self explicatory xD
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables to calculate log likelihood.
        lista_predictive (list): List of predictive variables.

    Returns:
        idata (InferenceData): An ArviZ InferenceData object containing the results of the HMC simulation.
    """
    rng = _np.random.default_rng(rng_seed)
    try:
        _os.mkdir(output_dir)
    except Exception as e:
        print('While making output_dir')
        print(e)
        pass

    json_filepath = output_dir + '/data.json'
    f = open(json_filepath, mode='w')
    def json_convert_helper(o):
        if isinstance(o, _np.int64): return int(o)
        elif isinstance(o, _np.ndarray): return list(o)
        else: return(o)
    _json.dump(data_dic, f, default=json_convert_helper)
#    json.dump(data_dic, f)
    f.close()

    try:
        _os.mkdir(output_dir + '/output_dir_HMC')
    except Exception as e:
        print('While making /output_dir_HMC')
        print(e)
        pass

    model = _cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(json_filepath, 4,4,
                       seed = rng.integers(1000000),
                       iter_warmup = iter_warmup,
                       iter_sampling = iter_sampling,
                       max_treedepth = max_treedepth,
                       inits = inits,
                       output_dir = output_dir + '/output_dir_HMC'
                       )


    try:
        _shutil.rmtree(output_dir + '/resHMC')
    except Exception as e:
        print('While removing /resHMC')
        print(e)
        pass

    fit.save_csvfiles(output_dir + '/resHMC')

    if len(lista_observed) == 0:
        idata = _az.from_cmdstan([output_dir + '/resHMC/' + file for file in _os.listdir(output_dir + '/resHMC')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                constant_data = json_filepath,
                                )
    else:
        idata = _az.from_cmdstan([output_dir + '/resHMC/' + file for file in _os.listdir(output_dir + '/resHMC')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                observed_data = json_filepath,
                                observed_data_var = lista_observed,
                                constant_data = json_filepath,
                                constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
                                log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
                                )
    
    return idata


def execute_pathfinder(data_dic,
                        rng_seed,
                        stan_file = './stan_codes/mean_estimation_prior.stan',
                        output_dir = './out',
                        draws = 1000,
                        inits = None,
                        history_size = None,
                        lista_observed = [],
                        lista_lklhood = [],
                        lista_predictive = [],
                        ):
    """
    Executes Pathfinder sampling for a given Stan model, with specified data, seed, and sampling parameters.

    Args:
        data_dic (dict): The data to be used in the model, in dictionary format.
        rng_seed (int): The seed for the random number generator.
        stan_file (str): The path to the Stan model file.
        output_dir (str): The directory where output files will be saved.
        draws (int): Number of approximate draws to return.
        inits (str or dict): Initialization values for model parameters.
        history_size (int): self explicatory xD
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables to calculate log likelihood.
        lista_predictive (list): List of predictive variables.

    Returns:
        idata (InferenceData): An ArviZ InferenceData object containing the results of the HMC simulation.
    """
    rng = _np.random.default_rng(rng_seed)
    try:
        _os.mkdir(output_dir)
    except Exception as e:
        print('While making output_dir')
        print(e)
        pass

    json_filepath = output_dir + '/data.json'
    f = open(json_filepath, mode='w')
    def json_convert_helper(o):
        if isinstance(o, _np.int64): return int(o)
        elif isinstance(o, _np.ndarray): return list(o)
        else: return(o)
    _json.dump(data_dic, f, default=json_convert_helper)
#    json.dump(data_dic, f)
    f.close()

    try:
        _os.mkdir(output_dir + '/output_dir_PF')
    except Exception as e:
        print('While making /output_dir_PF')
        print(e)
        pass

    model = _cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.pathfinder(json_filepath, draws=draws,
                           seed = rng.integers(1000000),
                           history_size = history_size,
                           inits = inits,
                           output_dir = output_dir + '/output_dir_PF'
                           )


    try:
        _shutil.rmtree(output_dir + '/resPF')
    except Exception as e:
        print('While removing /resPF')
        print(e)
        pass

    fit.save_csvfiles(output_dir + '/resPF')

    if len(lista_observed) == 0:
        idata = _az.from_cmdstan([output_dir + '/resPF/' + file for file in _os.listdir(output_dir + '/resPF')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                constant_data = json_filepath,
                                )
    else:
        idata = _az.from_cmdstan([output_dir + '/resPF/' + file for file in _os.listdir(output_dir + '/resPF')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                observed_data = json_filepath,
                                observed_data_var = lista_observed,
                                constant_data = json_filepath,
                                constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
                                log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
                                )    
    return idata


def load_HMC(output_dir = './out',
             lista_observed = [],
             lista_lklhood = [],
             lista_predictive = [],
             ):
    """
    Loads HMC simulation results from specified output directory and returns an ArviZ InferenceData object.

    Args:
        output_dir (str): The directory from which to load HMC simulation results.
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables for which log likelihood is calculated.
        lista_predictive (list): List of predictive variables.

    Returns:
        idata (InferenceData): An ArviZ InferenceData object containing the loaded HMC simulation results.
    """    

    json_filepath = output_dir + '/data.json'
    f = open(json_filepath, mode='r')
    data_dic = _json.load(f)
    f.close()

    if len(lista_observed) == 0:
        idata = _az.from_cmdstan([output_dir + '/resHMC/' + file for file in _os.listdir(output_dir + '/resHMC')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                constant_data = json_filepath,
                                )
    else:
        idata = _az.from_cmdstan([output_dir + '/resHMC/' + file for file in _os.listdir(output_dir + '/resHMC')],
                                posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                                observed_data = json_filepath,
                                observed_data_var = lista_observed,
                                constant_data = json_filepath,
                                constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
                                log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
                                )
    
    return idata




def execute_HMC_prior_posterior(data_dic,
                                rng_seed,
                                stan_dir = './stan_codes/',
                                stan_name = 'mean_estimation',
                                output_dir = './out',
                                iter_warmup_prior = 1000,
                                iter_sampling_prior = 1000,
                                iter_warmup = 1000,
                                iter_sampling = 1000,
                                inits = None,
                                max_treedepth = 10,
                                lista_observed = [],
                                lista_lklhood = [],
                                lista_predictive = [],
                                ):
    """
    Executes HMC sampling for both the prior and posterior distributions of a given Stan model, saving the results on the same inference data object

    Args:
        data_dic (dict): The data to be used in the model, in dictionary format.
        rng_seed (int): The seed for the random number generator.
        stan_dir (str): The directory containing Stan model files.
        stan_name (str): The base name of the Stan model files for prior and posterior.
        output_dir (str): The directory where output files will be saved.
        iter_warmup_prior (int): The number of warmup iterations for the prior.
        iter_sampling_prior (int): The number of sampling iterations for the prior.
        iter_warmup (int): The number of warmup iterations for the posterior.
        iter_sampling (int): The number of sampling iterations for the posterior.
        inits (str or dict): Initialization method or values for model parameters.
        max_treedepth (int): lmao
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables to calculate log likelihood.
        lista_predictive (list): List of predictive variables.

    Returns:
        idata (InferenceData): An ArviZ InferenceData object containing the results of both prior and posterior HMC simulations.
    """    

    rng = _np.random.default_rng(rng_seed)
    
    prior_stan_path = stan_dir + stan_name + '_prior.stan'
    posterior_stan_path = stan_dir + stan_name + '_posterior.stan'
    
    try:
        _os.mkdir(output_dir)
    except Exception as e:
        print('While making output_dir:')
        print(e)
        pass

    json_filepath = output_dir + '/data.json'
    f = open(json_filepath, mode='w')
    def json_convert_helper(o):
        if isinstance(o, _np.int64): return int(o)
        elif isinstance(o, _np.ndarray): return list(o)
        else: return(o)
    _json.dump(data_dic, f, default=json_convert_helper)
#    json.dump(data_dic, f)
    f.close()

    try:
        _os.mkdir(output_dir + '/output_dir_HMC_prior')
    except Exception as e:
        print('While making output_dir_HMC_prior')
        print(e)
        pass

    model = _cmdstanpy.CmdStanModel(stan_file=prior_stan_path)
    fit_prior = model.sample(json_filepath, 4,4,
                             seed = rng.integers(1000000),
                             iter_warmup = iter_warmup_prior,
                             iter_sampling = iter_sampling_prior,
                             max_treedepth = max_treedepth,
                             output_dir = output_dir + '/output_dir_HMC_prior'
                             )


    try:
        _shutil.rmtree(output_dir + '/resHMC_prior')
    except Exception as e:
        print('While deleting /resHMC_prior')
        print(e)
        pass

    fit_prior.save_csvfiles(output_dir + '/resHMC_prior')


    try:
        _os.mkdir(output_dir + '/output_dir_HMC_posterior')
    except Exception as e:
        print('While making output_dir_HMC_posterior')
        print(e)
        pass

    model = _cmdstanpy.CmdStanModel(stan_file=posterior_stan_path)
    fit_posterior = model.sample(json_filepath, 4,4,
                                 seed = rng.integers(1000000),
                                 iter_warmup = iter_warmup,
                                 iter_sampling = iter_sampling,
                                 max_treedepth = max_treedepth,
                                 output_dir = output_dir + '/output_dir_HMC_posterior'
                                 )


    try:
        _shutil.rmtree(output_dir + '/resHMC_posterior')
    except Exception as e:
        print('While deleting /resHMC_posterior')
        print(e)
        pass

    fit_posterior.save_csvfiles(output_dir + '/resHMC_posterior')



    
    idata = _az.from_cmdstan([output_dir + '/resHMC_posterior/' + file for file in _os.listdir(output_dir + '/resHMC_posterior')],
                            posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                            prior = [output_dir + '/resHMC_prior/' + file for file in _os.listdir(output_dir + '/resHMC_prior')],
                            prior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                            observed_data = json_filepath,
                            observed_data_var = lista_observed,
                            constant_data = json_filepath,
                            constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
                            log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
                            )
    
    return idata




def load_HMC_prior_posterior(stan_name = 'mean_estimation',
                             output_dir = './out',
                             lista_observed = [],
                             lista_lklhood = [],
                             lista_predictive = [],
                             ):
    """
    Loads HMC simulation results for both prior and posterior from specified output directory and returns an ArviZ InferenceData object.

    Args:
        stan_name (str): The base name of the Stan model files for prior and posterior.
        output_dir (str): The directory from which to load HMC simulation results for both prior and posterior.
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables for which log likelihood is calculated.
        lista_predictive (list): List of predictive variables.

    Returns:
        idata (InferenceData): An ArviZ InferenceData object containing the loaded results of both prior and posterior HMC simulations.
    """    
    
    json_filepath = output_dir + '/data.json'
    f = open(json_filepath, mode='r')
    data_dic = _json.load(f)
    f.close()
    
    idata = _az.from_cmdstan([output_dir + '/resHMC_posterior/' + file for file in _os.listdir(output_dir + '/resHMC_posterior')],
                            posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                            prior = [output_dir + '/resHMC_prior/' + file for file in _os.listdir(output_dir + '/resHMC_prior')],
                            prior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
                            observed_data = json_filepath,
                            observed_data_var = lista_observed,
                            constant_data = json_filepath,
                            constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
                            log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
                            )
    return idata





def get_expected_log_likelihood(idata, variables):
    """
    Calculate the expected logarithm of the likelihood of the data as 
    approximated by the empirical mean along the Monte Carlo sample. 
    This function uses the log_likelihood group from the input InferenceData object,
    and computes the sum of the logarithm of the likelihoods for the specified variables, across 
    all dimensions except `chain`, `draw`, or `sample`. 
    The log sum exp is computed for these sums, and log(N) is subtracted to obtain the expected log likelihood.

    Parameters
    ----------
    idata : az.InferenceData
        An ArviZ InferenceData object containing the posterior samples and log likelihood values.
        
    variables : list of str
        A list of variable names for which the expected log likelihood is computed.
        These variables should be present in the `log_likelihood` group of `idata`.

    Raises
    ------
    ValueError:
	If there is some inconsistency between the passed arguments

    Returns
    -------
    float
        The expected log likelihood value for the specified variables.

    Examples
    --------
    >>> idata = multitarget_latent_factor_execute_HMC(
            data_dic,
            rng.integers(1000000),
            X_test=true_vals_test_set_xr['X_test'].values
        )
    >>> get_expected_log_likelihood(idata, ['y_posterior'])
    -4849.833211101989
    >>> get_expected_log_likelihood(idata, ['y_prior'])
    -5160.257442177282

    Notes
    -----
    This function assumes that the input `idata` contains a `log_likelihood` group with log likelihood values
    for each variable specified in `variables`.
    The computation sums over all dimensions except for `chain`, `draw`, and `sample`,
    and then applies the log_sum_exp operation followed by subtracting log(N), where N is the number of summed elements. 
    This approach approximates the expected log likelihood by the empirical mean along the Monte Carlo samples.

    References
    ----------
    If the function is based on published research or algorithms, cite these sources.
    """
    
    # Check if idata is an ArviZ InferenceData object
    if not isinstance(idata, _az.InferenceData):
        raise ValueError("idata must be an ArviZ InferenceData object.")
    
    # Check if log_likelihood group exists in idata
    if 'log_likelihood' not in idata.groups():
        raise ValueError("The input idata does not contain a 'log_likelihood' group.")
    
    # Check if variables is a list of strings
    if not isinstance(variables, list) or not all(isinstance(var, str) for var in variables):
        raise ValueError("variables must be a list of strings.")
    
    # Check if all variables are present in the log_likelihood group of idata
    log_likelihood_vars = list(idata.log_likelihood.data_vars)
    missing_vars = [var for var in variables if var not in log_likelihood_vars]
    if missing_vars:
        raise ValueError(f"Variables {missing_vars} are not present in the 'log_likelihood' group of idata.")
    
    # Check for consistent dimension names across specified variables
    if( len(set([idata.log_likelihood[var].dims[0] for var in variables])) != 1 ):
        raise ValueError("Inconsistent dimension names found across specified variables. Ensure consistency in the use of 'chain', 'draw', and 'sample' dimensions.")
    
    log_lklhood_dataset = idata.log_likelihood[variables]
    aux = \
    _np.array(
        [
            log_lklhood_dataset[lglklhd].sum(
                [dim for dim in log_lklhood_dataset[lglklhd].dims if dim not in ['chain','draw','sample']]
            ) for lglklhd in log_lklhood_dataset
        ]
    ).sum(axis=0).flatten()

    return(_sps.logsumexp(aux) - _np.log(len(aux)))



def MAP_and_LaplaceSample(data_dic,
                          rng_seed,
                          stan_file = './stan_codes/mean_estimation_posterior.stan',
                          output_dir = './out',
                          number_of_maximum_samples = 32,
                          max_iter_max_samples = 20,
                          inits = None,
                          lista_observed = ['X'],
                          lista_lklhood = ['X'],
                          lista_predictive = ['X'],
                          laplace_draws = 100000,
                          ):
    """
    Finds the Maximum A Posteriori (MAP) estimates and performs Laplace approximation sampling for a given Stan model.

    Args:
        data_dic (dict): The data to be used in the model, in dictionary format.
        rng_seed (int): The seed for the random number generator.
        stan_file (str): The path to the Stan model file.
        output_dir (str): The directory where output files will be saved.
        number_of_maximum_samples (int): The number of MAP samples to generate.
        max_iter_max_samples (int): Maximum number of iterations to attempt generating each MAP sample.
        inits (str or dict): Initialization method or values for model parameters.
        lista_observed (list): List of observed data variables.
        lista_lklhood (list): List of variables to calculate log likelihood.
        lista_predictive (list): List of predictive variables.
        laplace_draws (int): The number of samples to generate using Laplace approximation.

    Returns:
        dict: A dictionary containing 'MAP_inference_data' and 'Laplace_inference_data', each an ArviZ InferenceData object.
    """
    
    rng = _np.random.default_rng(rng_seed)
    
    try:
        _shutil.rmtree(output_dir)
    except Exception as e:
        print('While deleting output_dir')
        print(e)
        pass

    try:
        _os.mkdir(output_dir)
    except Exception as e:
        print('While creating output_dir')
        print(e)
        pass

    opt_out_dir = output_dir + '/MAPs'
    opt_out_dir_saved = output_dir + '/SavedMAPs'
    opt_out_ML_MAPs = output_dir + '/ML_MAP'

    try:
        _os.mkdir(opt_out_dir)
    except Exception as e:
        print('While creating /MAPs')
        print(e)
        pass

    try:
        _os.mkdir(opt_out_dir_saved)
    except Exception as e:
        print('While creating /SavedMAPs')
        print(e)
        pass

    try:
        _os.mkdir(opt_out_ML_MAPs)
    except Exception as e:
        print('While creating /ML_MAP')
        print(e)
        pass


    _os.mkdir(output_dir + '/MAP_data_block')
    json_filepath = output_dir + '/MAP_data_block/data.json'
    f = open(json_filepath, mode='w')
    def json_convert_helper(o):
        if isinstance(o, _np.int64): return int(o)
        elif isinstance(o, _np.ndarray): return list(o)
        else: return(o)
    _json.dump(data_dic, f, default=json_convert_helper)
    f.close()

    model = _cmdstanpy.CmdStanModel(stan_file=stan_file)

    for i in range(number_of_maximum_samples):
        tries = 0
        while True:
            try:
                fit = model.optimize(data_dic, output_dir=opt_out_dir, seed=rng.integers(1000000), inits=inits, jacobian=True)
            except Exception as e:
                print(e)
                if tries >= max_iter_max_samples:
                    raise Exception('Failed to compute Maximum a Posteriori')
                else:
                    tries += 1
                    continue
            else:
                fit.save_csvfiles(opt_out_dir_saved)
                nome_file = str(fit).split('csv_file:\n')[1].split('\n')[0].split('/')[-1]
                path_file = opt_out_dir_saved + '/' + nome_file
                _os.rename(path_file, opt_out_dir_saved + '/MaximumAPosteriori_sampleNo_' + str(i) + '.csv' )
                break
    

    lista_maps = sorted( [opt_out_dir_saved + '/' + file for file in _os.listdir(opt_out_dir_saved) if file[-4:] == '.csv'] )

    optIData = _az.from_cmdstan(
        lista_maps,
        log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
    )


    def _get_posterior_lklhood(idata):
        rep = 0
        for var in idata.log_likelihood.data_vars:
            aux = idata.log_likelihood[var].copy()
            while True:
                try:
                    aux2 = aux.sum(axis=2)
                    aux = aux2
                except:
                    break
            if rep == 0:
                log_lklhood = aux
            else:
                log_lklhood += aux
            rep += 1

        return(log_lklhood)
    
    bestNo = _np.argmax( _get_posterior_lklhood( optIData )[:,0].values )
    
    _os.rename(lista_maps[bestNo], opt_out_ML_MAPs + '/' + 'MAP_with_ML.csv')

    try:
        _shutil.rmtree(opt_out_dir)
    except Exception as e:
        print('While deleting /MAPs')
        print(e)
        pass

    try:
        _shutil.rmtree(opt_out_dir_saved)
    except Exception as e:
        print('While deleting /SavedMAPs')
        print(e)
        pass    

    model.laplace_sample(data_dic,
                         mode = opt_out_ML_MAPs + '/' + 'MAP_with_ML.csv',
                         jacobian=True,
                         draws=laplace_draws,
                         seed = rng.integers(1000000),
                         output_dir=output_dir + '/LaplaceSample' )

    optIData =\
    _az.from_cmdstan(
        [output_dir + '/ML_MAP/' + file for file in _os.listdir(output_dir + '/ML_MAP') if file[-4:] == '.csv'],
        posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
        observed_data = json_filepath,
        observed_data_var = lista_observed,
        constant_data = json_filepath,
        constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
        log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
    )

    laplaceIData =\
    _az.from_cmdstan(
        [output_dir + '/LaplaceSample/' + file for file in _os.listdir(output_dir + '/LaplaceSample') if file[-4:] == '.csv'],
        posterior_predictive = [var + '_predictive' for var in lista_predictive] if not lista_predictive is None else None,
        observed_data = json_filepath,
        observed_data_var = lista_observed,
        constant_data = json_filepath,
        constant_data_var = [var for var in list(data_dic.keys()) if var not in lista_observed],
        log_likelihood = {var: 'log_lik_'+var for var in lista_lklhood}
    )
    
    return(
        {
            'MAP_inference_data': optIData,
            'Laplace_inference_data': laplaceIData,
        }
    )
