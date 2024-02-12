

import numpy as _np
import scipy.special as _sps
import pandas as _pd
import arviz as _az
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import seaborn as _sns
import os as _os
import shutil as _shutil
import xarray as _xr

import scipy.stats as _spst
from itertools import product
import skfda.representation.basis as scikit_basis_funs
import xarray_einstats as _xrein



# interactive plotting tools
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual



from .HMC_helper import execute_HMC as _execute_HMC
from .HMC_helper import MAP_and_LaplaceSample

import pkg_resources


__all__ = ['sample_from_prior', 'renaming_convention', 'default_basis_dictionary_builder', 'compute_BLambaBetaX', 'sample_from_posterior',
            'sample_from_posterior_predictive', 'dataset_generator', 'one_functional_target_dictionary_builder',
            'sample_conditional_predictive', 'sample_unconditional_predictive',
            'get_relationship_between_targets','Varimax_RSP', 'sample_projection_on_varimaxed_space']


def sample_from_prior(data_dic, X_test, rng_seed, n_samples):
    """
    This is part of the functions relating to the analysis of multitarget latent factor model.
    This function samples from the model described in the Notes section and returns both
    the prior samples and prior predictive samples.

    The prior predictive samples are produced both for the covariates X in the training set
    in the data_dictionary and for the covariates X in the test set in X_test.

    Parameters
    ----------
    data_dic : dict
        A dictionary containing all the data/hyperparameters of the model, which means
    	a dictionary with the entries describe in the Notes section.
    	
    X_test : np.ndarray
    	A 2D numpy array of floats representing the covariates of the test set.
        It's dimensionality should be:
		X_test.shape[0] == data_dic['r']
		X_test.shape[1] is the number of samples in the test set

    rng_seed : int
	Initialized the random number generator used inside the function

    n_samples : int
	The number of draws to sample from the prior

    Raises
    ------
    ValueError
	If data_dic does not contain all the required keys, or there is any
	violation in the rules that the hyperparameters should follow

    Returns
    -------
    xr.Dataset
	An xarray dataset containing all prior, prior predictive and prior predictions

    Notes
    -----
    Model:
    	pass

    data_dic : dict
    	A dictionary containing the following keys:
	N		: (Positive integer) The number of samples in the training set
	L1		: (Positive integer) The number of positions in which the first  target has been sampled
	L2		: (Positive integer) The number of positions in which the second target has been sampled
	p1		: (Positive integer) The number of basis functions used to describe the first  target
	p2		: (Positive integer) The number of basis functions used to describe the second target
	k		: (Positive integer) [Hyperparameter] The number of latent factors to be fitted
	r		: (Positive integer) The number of covriates used in regression
	t1		: (vector[L1]) The locations at which the observations have been made for the first  target
	y1		: (L1 x N real matrix) The observations for the first  target at the locations described by t1
	t2		: (vector[L2]) The locations at which the observations have been made for the second target
	y2		: (L2 x N real matrix) The observations for the second target at the locations described by t2
	X		: (r x N real matrix) The covariates
	B1		: (L1 x p1 real matrix) The matrix describing the basis functions used for the first  target (y1 ~ B1 x θ1)
	B2		: (L2 x p2 real matrix) The matrix describing the basis functions used for the second target (y2 ~ B2 x θ2)
	v		: (Positive real number) [Hyperparameter] The degrees of freedom for the prior of the Latent Factors (Λ)
	nu		: (Positive real number) [Hyperparameter] The degrees of freedom for the prior of the Regression Coefficients (β)
	alpha_psi	: (Positive real number) [Hyperparameter] The shape parameter for ψ (~ Gamma(α_ψ,β_ψ))
	beta_psi	: (Positive real number) [Hyperparameter] The rate  parameter for ψ (~ Gamma(α_ψ,β_ψ))

	psi_noise_scale_multiplier_1 : 
					(Positive real number)
					[Hyperparameter] Everytime ψ appears in the model, it is multiplied by this factor (to improve sampling)

	psi_noise_scale_multiplier_2 :
					(Positive real number)
					[Hyperparameter] Everytime ψ appears in the model, it is multiplied by this factor (to improve sampling)

	alpha_sigma	: (Positive real number) [Hyperparameter] The shape paraemter for τ_θ (~ Gamma(α_σ,β_σ))
        beta_sigma	: (Positive real number) [Hyperparameter] The rate paraemter for τ_θ (~ Gamma(α_σ,β_σ))

	theta_noise_scale_multiplier_1 : 
					(Positive real number)
					[Hyperparameter] Everytime sqrt(1/τ_θ) appears in the model, it is multiplied by this factor (to improve sampling)

	theta_noise_scale_multiplier_2 : 
					(Positive real number)
					[Hyperparameter] Everytime sqrt(1/τ_θ) appears in the model, it is multiplied by this factor (to improve sampling)

	alpha_1		: (Positive real number) [Hyperparameter] The shape parameter for (τ_λ)_1 (~ Gamma(α_1, 1))
	alpha_2		: (Positive real number) [Hyperparameter] The shape parameter for δ (~ Gamma(α_2, 1))

    	Note: Additional keys may be present, but are not used by this function.
    
    Examples
    --------
    Provide examples of how to use the function.

    >>> function_name(value1, value2)
    expected_result

    References
    ----------
    If the function is based on published research or algorithms, cite these sources.

    See Also
    --------
    Mention related functions or classes here.

    """
    
    # Validate data_dic
    required_keys = ['N', 'L1', 'L2', 'p1', 'p2', 'k', 'r', 't1', 'y1', 't2', 'y2', 'X', 'B1', 'B2', 'v', 'nu', 'alpha_psi', 'beta_psi', 'psi_noise_scale_multiplier_1', 'psi_noise_scale_multiplier_2', 'alpha_sigma', 'beta_sigma', 'theta_noise_scale_multiplier_1', 'theta_noise_scale_multiplier_2', 'alpha_1', 'alpha_2']
    for key in required_keys:
        if key not in data_dic:
            raise ValueError(f"Missing required data_dic key: {key}")
    
    # Validate shapes and types of data_dic values
    # Validate N - Positive integer
    if not isinstance(data_dic['N'], (int, _np.int_)) or data_dic['N'] <= 0:
        raise ValueError("N must be a positive integer")

    # Validate L1, L2, p1, p2, k, r - Positive integers
    for key in ['L1', 'L2', 'p1', 'p2', 'k', 'r']:
        if not isinstance(data_dic[key], (int, _np.int_)) or data_dic[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    # Validate t1, t2 - 1D numpy arrays of floats
    for key in ['t1', 't2']:
        if not isinstance(data_dic[key], _np.ndarray) or data_dic[key].ndim != 1 or data_dic[key].dtype.kind not in 'if':
            raise ValueError(f"{key} must be a 1D numpy array of floats")

    # Validate y1, y2 - 2D numpy arrays of floats
    for key in ['y1', 'y2']:
        if not isinstance(data_dic[key], _np.ndarray) or data_dic[key].ndim != 2 or data_dic[key].dtype.kind not in 'if':
            raise ValueError(f"{key} must be a 2D numpy array of floats")

    # Validate X - 2D numpy array of floats with dimensions (r, N)
    if not isinstance(data_dic['X'], _np.ndarray) or data_dic['X'].shape != (data_dic['r'], data_dic['N']):
        raise ValueError("X must be a 2D numpy array with dimensions (r, N)")

    # Validate B1, B2 - 2D numpy arrays of floats
    for key in ['B1', 'B2']:
        if not isinstance(data_dic[key], _np.ndarray) or data_dic[key].ndim != 2 or data_dic[key].dtype.kind not in 'if':
            raise ValueError(f"{key} must be a 2D numpy array of floats")

    # Validate hyperparameters (v, nu, alpha_psi, beta_psi, alpha_sigma, beta_sigma, alpha_1, alpha_2) - Positive reals
    for key in ['v', 'nu', 'alpha_psi', 'beta_psi', 'alpha_sigma', 'beta_sigma', 'alpha_1', 'alpha_2']:
        if not (isinstance(data_dic[key], (float, _np.float_)) or isinstance(data_dic[key], (int, _np.int_))) or data_dic[key] <= 0:
            raise ValueError(f"{key} must be a positive real number")

    # Validate noise scale multipliers (psi_noise_scale_multiplier_1, psi_noise_scale_multiplier_2,
    # theta_noise_scale_multiplier_1, theta_noise_scale_multiplier_2) - Positive reals
    for key in ['psi_noise_scale_multiplier_1', 'psi_noise_scale_multiplier_2', 'theta_noise_scale_multiplier_1', 'theta_noise_scale_multiplier_2']:
        if not (isinstance(data_dic[key], (float, _np.float_)) or isinstance(data_dic[key], (int, _np.int_))) or data_dic[key] <= 0:
            raise ValueError(f"{key} must be a positive real number")

    # X_test
    if not isinstance(X_test, _np.ndarray) or X_test.ndim != 2 or X_test.shape[0] != data_dic['r']:
        raise ValueError("X_test must be a 2D numpy array with the number of rows equal to data_dic['r']")
    
    # rng_seed
    if not isinstance(rng_seed, (int, _np.int_)):
        raise ValueError("rng_seed must be an integer")

    # n_samples
    if not isinstance(n_samples, (int, _np.int_)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    
    # Initialize the random number generator
    rng = _np.random.default_rng(rng_seed)

    # Get the covariates in the training set
    X = data_dic['X']
    
    # Pick all data that could be needed
    number_test_samples = X_test.shape[1]
    k = data_dic['k']
    r = data_dic['r']
    p1 = data_dic['p1']
    p2 = data_dic['p2']
    L1 = data_dic['L1']
    L2 = data_dic['L2']
    B1 = data_dic['B1']
    B2 = data_dic['B2']
    N = data_dic['N']

    # Sample from τ_λ
    δ1 = _np.zeros((n_samples, k))
    δ1[:,0] = _spst.gamma(data_dic['alpha_1'], loc=0, scale=1/1).rvs(random_state=rng, size=(n_samples, ))
    δ1[:,1:k] = _spst.gamma(data_dic['alpha_2'], loc=0, scale=1/1).rvs(random_state=rng, size=(n_samples, k-1))
    δ2 = _np.zeros((n_samples, k))
    δ2[:,0] = _spst.gamma(data_dic['alpha_1'], loc=0, scale=1/1).rvs(random_state=rng, size=(n_samples, ))
    δ2[:,1:k] = _spst.gamma(data_dic['alpha_2'], loc=0, scale=1/1).rvs(random_state=rng, size=(n_samples, k-1))
    τ_λ1 = _np.cumprod(δ1, axis=1)
    τ_λ2 = _np.cumprod(δ2, axis=1)

    # Sample from τ_θ
    τ_θ1 = _spst.gamma(data_dic['alpha_sigma'], loc=0, scale=1/data_dic['beta_sigma']).rvs(random_state=rng, size=(n_samples, p1))
    τ_θ2 = _spst.gamma(data_dic['alpha_sigma'], loc=0, scale=1/data_dic['beta_sigma']).rvs(random_state=rng, size=(n_samples, p2))

    # Sample from ψ
    ψ1 = _np.sqrt(
        _spst.invgamma(data_dic['alpha_psi'], loc=0, scale=data_dic['beta_psi']).rvs(random_state=rng, size=(n_samples, ))
    )
    ψ2 = _np.sqrt(
        _spst.invgamma(data_dic['alpha_psi'], loc=0, scale=data_dic['beta_psi']).rvs(random_state=rng, size=(n_samples, ))
    )

    # Sample from β
    β = _spst.t(data_dic['nu'], 0,1).rvs(random_state=rng, size=(n_samples, k,r))
    
    # Sample from Λ | τ_λ
    # multiply each sampled column by the approriate component of τ_λ
    Λ1 = (
        _xr.DataArray(_spst.t(data_dic['v'], 0,1).rvs(random_state=rng, size=(n_samples, p1,k)), dims=['sample','basis_fun','latent_factor'])*\
        _np.sqrt(1/_xr.DataArray(τ_λ1, dims=['sample','latent_factor']))
    ).values
    Λ2 = (
        _xr.DataArray(_spst.t(data_dic['v'], 0,1).rvs(random_state=rng, size=(n_samples, p2,k)), dims=['sample','basis_fun','latent_factor'])*\
        _np.sqrt(1/_xr.DataArray(τ_λ2, dims=['sample','latent_factor']))
    ).values

    regr_coeffs1 = _np.zeros((n_samples, L1, r))
    regr_coeffs2 = _np.zeros((n_samples, L2, r))
    estimated_y1 = _np.zeros((n_samples, L1, N))
    estimated_y2 = _np.zeros((n_samples, L2, N))
    estimated_y1_test = _np.zeros((n_samples, L1, number_test_samples))
    estimated_y2_test = _np.zeros((n_samples, L2, number_test_samples))
    Σ11 = _np.zeros((n_samples, L1,L1))
    Σ22 = _np.zeros((n_samples, L2,L2))
    Σ12 = _np.zeros((n_samples, L1,L2))
    log_lik_y = _np.zeros((n_samples, N))
    y1_predictive = _np.zeros((n_samples, L1,N))
    y2_predictive = _np.zeros((n_samples, L2,N))
    y1_test_predictive = _np.zeros((n_samples, L1,number_test_samples))
    y2_test_predictive = _np.zeros((n_samples, L2,number_test_samples))
    
    for sample_idx in range(n_samples):
        # Compute some useful generated quantities (means)
        regr_coeffs1[sample_idx] = B1 @ Λ1[sample_idx] @ β[sample_idx]
        regr_coeffs2[sample_idx] = B2 @ Λ2[sample_idx] @ β[sample_idx]
        estimated_y1[sample_idx] = regr_coeffs1[sample_idx] @ X
        estimated_y2[sample_idx] = regr_coeffs2[sample_idx] @ X
        estimated_y1_test[sample_idx] = regr_coeffs1[sample_idx] @ X_test
        estimated_y2_test[sample_idx] = regr_coeffs2[sample_idx] @ X_test

        # Compute other useful generated quantities (covariance structure)
        # This is the covariance structure that one gets for
        # (y1, y2) | Λ1, Λ2, β, τ_λ1, τ_λ2, τ_θ1, τ_θ2, ψ1, ψ2
        # (remember that B1 and B2 are fixed)
        Σ11[sample_idx] = assemble_Σ11_22(
            B1, Λ1[sample_idx],
            _np.square(data_dic['theta_noise_scale_multiplier_1'])/(τ_θ1[sample_idx]),
            ψ1[sample_idx]*data_dic['psi_noise_scale_multiplier_1']
        )
        Σ22[sample_idx] = assemble_Σ11_22(
            B2, Λ2[sample_idx],
            _np.square(data_dic['theta_noise_scale_multiplier_2'])/(τ_θ2[sample_idx]),
            ψ2[sample_idx]*data_dic['psi_noise_scale_multiplier_2']
        )
        Σ12[sample_idx] = assemble_Σ12(B1, Λ1[sample_idx], B2, Λ2[sample_idx])
        
        Σ = assemble_Σ(Σ11[sample_idx], Σ22[sample_idx], Σ12[sample_idx])
        cov_obj_Σ = _spst.Covariance.from_cholesky( _np.linalg.cholesky(Σ) )
        
        # Compute log likelihoods and prior predictives
        y_predictive = _np.zeros((L1+L2, N))
        
        for i in range(N):
            estimated_y_aux = _np.zeros(L1+L2)
            estimated_y_aux[0:L1] = estimated_y1[sample_idx][:,i]
            estimated_y_aux[L1:(L1+L2)] = estimated_y2[sample_idx][:,i]
            
            MVN = _spst.multivariate_normal( mean = estimated_y_aux, cov = cov_obj_Σ )
            log_lik_y[sample_idx][i] = MVN.logpdf(_np.concatenate([data_dic['y1'], data_dic['y2']])[:,i])
            
            y_predictive_aux = MVN.rvs(random_state = rng)[0]

            y1_predictive[sample_idx][:,i] = y_predictive_aux[0:L1]
            y2_predictive[sample_idx][:,i] = y_predictive_aux[L1:(L1+L2)]
            
        y_test_predictive = _np.zeros((L1+L2, number_test_samples))
            
        for i in range(number_test_samples):
            estimated_y_aux = _np.zeros(L1+L2)
            estimated_y_aux[0:L1] = estimated_y1_test[sample_idx][:,i]
            estimated_y_aux[L1:(L1+L2)] = estimated_y2_test[sample_idx][:,i]
                
            MVN = _spst.multivariate_normal( mean = estimated_y_aux, cov = cov_obj_Σ )
                
            y_predictive_aux = MVN.rvs(random_state = rng)[0]
        
            y1_test_predictive[sample_idx][:,i] = y_predictive_aux[0:L1]
            y2_test_predictive[sample_idx][:,i] = y_predictive_aux[L1:(L1+L2)]
        
    return(
        _xr.Dataset(
            {
                'Lambda1': (('sample','Lambda1_dim_0','Lambda1_dim_1'), Λ1),
                'Lambda2': (('sample','Lambda2_dim_0','Lambda2_dim_1'), Λ2),
                'beta': (('sample','beta_dim_0','beta_dim_1'), β),
                'tau_theta1': (('sample','tau_theta1_dim_0'), τ_θ1),
                'tau_theta2': (('sample','tau_theta2_dim_0'), τ_θ2),
                'delta1': (('sample','delta1_dim_0'), δ1),
                'delta2': (('sample','delta2_dim_0'), δ2),
                'psi1': (('sample',), ψ1),
                'psi2': (('sample',), ψ2),
                'tau_lambda1': (('sample','tau_lambda1_dim_0'), τ_λ1),
                'tau_lambda2': (('sample','tau_lambda2_dim_0'), τ_λ2),
                'regr_coeffs1': (('sample','regr_coeffs1_dim_0','regr_coeffs1_dim_1'), regr_coeffs1),
                'regr_coeffs2': (('sample','regr_coeffs2_dim_0','regr_coeffs2_dim_1'), regr_coeffs2),
                'estimated_y1': (('sample','estimated_y1_dim_0','estimated_y1_dim_1'), estimated_y1),
                'estimated_y2': (('sample','estimated_y2_dim_0','estimated_y2_dim_1'), estimated_y2),
                'Sigma_11': (('sample','Sigma_11_dim_0','Sigma_11_dim_1'), Σ11),
                'Sigma_22': (('sample','Sigma_22_dim_0','Sigma_22_dim_1'), Σ22),
                'Sigma_12': (('sample','Sigma_12_dim_0','Sigma_12_dim_1'), Σ12),
                'Sigma_21': (('sample','Sigma_21_dim_0','Sigma_21_dim_1'), _np.transpose(Σ12, axes=(0,2,1))),
                'log_lik_y': (('sample','log_lik_y_dim_0'), log_lik_y),
                'y1_predictive': (('sample','y1_predictive_dim_0','y1_predictive_dim_1'), y1_predictive),
                'y2_predictive': (('sample','y2_predictive_dim_0','y2_predictive_dim_1'), y2_predictive),
                'y1_test_predictive': (('sample','target_1_dim_idx','test_sample_idx'), y1_test_predictive),
                'y2_test_predictive': (('sample','target_2_dim_idx','test_sample_idx'), y2_test_predictive),
            }
        )
    )    




def assemble_Σ11_22(B, Λ, var_θ, ψ):
    """
    Assembles the Covariance structure of
    y1/y2 | Λ1, Λ2, β, τ_λ1, τ_λ2, τ_θ1, τ_θ2, ψ1, ψ2
    according to the model provided in
    `sample_from_prior`.

    Parameters
    ----------
    B : np.ndarray 
	A 2D numpy array of floats describing the basis matrix.
	The array should have shape 			(n_sampling_points, n_basis_functions)
	or with the naming convention of the model 	(L,p)
    Λ : np.ndarray
	A 2D numpy array of floats describing the latent factors.
	The array should have shape 			(n_basis_functions, n_latent_factors)
	or with the naming convention of the model 	(p,k)
    var_θ : np.ndarray
	A 1D numpy array of floats describing the heteroschedastic variances of the θ's (scores of the basis functions).
	This is computed as:
		np.square(data_dic['theta_noise_scale_multiplier_<idx>'])/τ_θ<idx>
	The array should have shape 			(n_basis_functions, )
	or with the naming convention of the model 	(p, )
    ψ : float
	A float describing the standard deviation of the white noise of the observations about the reconstructed functions.
	This is computed as:
	        ψ<idx>*data_dic['psi_noise_scale_multiplier_<idx>']

    Raises
    ------
    ValueError
	If the matrices do not have compatible dimensions or var_θ or ψ are negative

    Returns
    -------
    np.ndarray
	A 2D numpy array of floats containing Σ11 or Σ22
	The array will have shape 			(n_sampling_points, n_sampling_points)
	or with the naming convention of the model 	(L,L)

    See Also
    --------
    assemble_Σ12
    assemble_Σ

    """

    if not isinstance(B, _np.ndarray) or B.ndim != 2:
        raise ValueError("B must be a 2D numpy array")
    if not isinstance(Λ, _np.ndarray) or Λ.ndim != 2:
        raise ValueError("Λ must be a 2D numpy array")
    if not isinstance(var_θ, _np.ndarray) or var_θ.ndim != 1:
        raise ValueError("var_θ must be a 1D numpy array")
    if not (isinstance(ψ, (float, _np.float_)) or isinstance(ψ, (int, _np.int_))) or ψ <= 0:
        raise ValueError("ψ must be a positive real number")
    if B.shape[1] != Λ.shape[0]:
        raise ValueError("Incompatible dimensions for n_basis_functions")
    if B.shape[1] != var_θ.size:
        raise ValueError("Incompatible dimensions for n_basis_functions")
    if (var_θ < 0).any():
        raise ValueError("Negative Variance in var_θ")
    
    # Define an useful function
    def _add_diag(matrix, vector):
        aux = _np.diag(matrix)
        aux = aux + vector
        res = matrix.copy()
        _np.fill_diagonal(res, aux)
        return(res)

    L = B.shape[0]

    return(
        _add_diag(
            (B) @ ( _add_diag( (Λ) @ (Λ.T) , var_θ ) ) @ (B.T),
            _np.repeat(_np.square(ψ), L )
        )
    )
    
def assemble_Σ12(B1, Λ1, B2, Λ2):
    """
    Assembles the Covariance structure of
    y1/y2 | Λ1, Λ2, β, τ_λ1, τ_λ2, τ_θ1, τ_θ2, ψ1, ψ2
    according to the model provided in
    `sample_from_prior`.

    Parameters
    ----------
    B1 : np.ndarray 
	A 2D numpy array of floats describing the basis matrix.
	The array should have shape 			(n_sampling_points_1, n_basis_functions_1)
	or with the naming convention of the model 	(L1,p1)
    Λ1 : np.ndarray
	A 2D numpy array of floats describing the latent factors.
	The array should have shape 			(n_basis_functions_1, n_latent_factors)
	or with the naming convention of the model 	(p1,k)
    B2 : np.ndarray 
	A 2D numpy array of floats describing the basis matrix.
	The array should have shape 			(n_sampling_points_2, n_basis_functions_2)
	or with the naming convention of the model 	(L2,p2)
    Λ2 : np.ndarray
	A 2D numpy array of floats describing the latent factors.
	The array should have shape 			(n_basis_functions_2, n_latent_factors)
	or with the naming convention of the model 	(p2,k)

    Raises
    ------
    ValueError
	If the matrices do not have compatible dimensions

    Returns
    -------
    np.ndarray
	A 2D numpy array of floats containing Σ12
	The array will have shape 			(n_sampling_points_1, n_sampling_points_2)
	or with the naming convention of the model 	(L1,L2)

    See Also
    --------
    assemble_Σ11_Σ22
    assemble_Σ

    """
    
    if not isinstance(B1, _np.ndarray) or B1.ndim != 2:
        raise ValueError("B1 must be a 2D numpy array")
    if not isinstance(B2, _np.ndarray) or B2.ndim != 2:
        raise ValueError("B2 must be a 2D numpy array")
    if not isinstance(Λ1, _np.ndarray) or Λ1.ndim != 2:
        raise ValueError("Λ1 must be a 2D numpy array")
    if not isinstance(Λ2, _np.ndarray) or Λ2.ndim != 2:
        raise ValueError("Λ2 must be a 2D numpy array")
    if B1.shape[1] != Λ1.shape[0]:
        raise ValueError("Incompatible dimensions for n_basis_functions_1")
    if B2.shape[1] != Λ2.shape[0]:
        raise ValueError("Incompatible dimensions for n_basis_functions_2")
    if Λ1.shape[1] != Λ2.shape[1]:
        raise ValueError("Incompatible dimensions for n_latent_factors")

    return(
        B1 @ Λ1 @ (Λ2.T) @ (B2.T)
    )
    


def assemble_Σ(Σ11, Σ22, Σ12):
    """
    Assembles the Covariance structure of
    (y1, y2) | Λ1, Λ2, β, τ_λ1, τ_λ2, τ_θ1, τ_θ2, ψ1, ψ2
    according to the model provided in
    `sample_from_prior`, given the provided blocks.

    Parameters
    ----------
    Σ11 : np.ndarray 
	A 2D numpy array of floats
	The array should have shape 			(n_sampling_points_1, n_sampling_points_1)
	or with the naming convention of the model 	(L1,L1)
    Σ22 : np.ndarray 
	A 2D numpy array of floats
	The array should have shape 			(n_sampling_points_2, n_sampling_points_2)
	or with the naming convention of the model 	(L2,L2)
    Σ12 : np.ndarray 
	A 2D numpy array of floats
	The array should have shape 			(n_sampling_points_1, n_sampling_points_2)
	or with the naming convention of the model 	(L1,L2)

    Raises
    ------
    ValueError
	If the matrices do not have compatible dimensions

    Returns
    -------
    np.ndarray
	A 2D numpy array of floats containing Σ
	The array will have shape 			(n_sampling_points_1 + n_sampling_points_2, n_sampling_points_1 + n_sampling_points_2)
	or with the naming convention of the model 	(L1 + L2, L1 + L2)

    Notes
    -----
    Σ is assembled as follows
            Σ = [  Σ11   Σ12 ]
                [ Σ12.T  Σ22 ]

    See Also
    --------
    assemble_Σ11_Σ22
    assemble_Σ12

    """

    if not isinstance(Σ11, _np.ndarray) or Σ11.ndim != 2 or Σ11.shape[0] != Σ11.shape[1]:
        raise ValueError("Σ11 must be a square 2D numpy array")
    if not isinstance(Σ22, _np.ndarray) or Σ22.ndim != 2 or Σ22.shape[0] != Σ22.shape[1]:
        raise ValueError("Σ22 must be a square 2D numpy array")
    if not isinstance(Σ12, _np.ndarray) or Σ12.ndim != 2:
        raise ValueError("Σ12 must be a 2D numpy array")
    if Σ11.shape[0] != Σ12.shape[0]:
        raise ValueError("Incompatible shapes between Σ11 and Σ12")
    if Σ22.shape[0] != Σ12.shape[1]:
        raise ValueError("Incompatible shapes between Σ22 and Σ12")
    
    L1 = Σ11.shape[0]
    L2 = Σ22.shape[0]
    
    Σ = _np.zeros((L1+L2,L1+L2))
    Σ[0:L1,0:L1] = Σ11
    Σ[L1:(L1+L2),0:L1] = Σ12.T
    Σ[0:L1,L1:(L1+L2)] = Σ12
    Σ[L1:(L1+L2),L1:(L1+L2)] = Σ22

    return(Σ)


def renaming_convention(xr_datum):
    """
    A useful function to make the xarray dimensions coherent when performing numerical operations
    """
    
    rename_dict = \
    {
        'theta1_dim_0':'basis_fun_branch_1_idx',
        'theta1_dim_1':'sample_idx',
        'theta2_dim_0':'basis_fun_branch_2_idx',
        'theta2_dim_1':'sample_idx',
        'Lambda1_dim_0':'basis_fun_branch_1_idx',
        'Lambda1_dim_1':'latent_factor_idx',
        'Lambda2_dim_0':'basis_fun_branch_2_idx',
        'Lambda2_dim_1':'latent_factor_idx',
        'eta_dim_0':'latent_factor_idx',
        'eta_dim_1':'sample_idx',
        'beta_dim_0':'latent_factor_idx',
        'beta_dim_1':'covariate_idx',
        'tau_theta1_dim_0':'basis_fun_branch_1_idx',
        'tau_theta2_dim_0':'basis_fun_branch_2_idx',
        'delta1_dim_0':'latent_factor_idx',
        'delta2_dim_0':'latent_factor_idx',
        'tau_lambda1_dim_0':'latent_factor_idx',
        'tau_lambda2_dim_0':'latent_factor_idx',
        'regr_coeffs1_dim_0':'target_1_dim_idx',
        'regr_coeffs1_dim_1':'covariate_idx',
        'regr_coeffs2_dim_0':'target_2_dim_idx',
        'regr_coeffs2_dim_1':'covariate_idx',
        'estimated_y1_dim_0':'target_1_dim_idx',
        'estimated_y1_dim_1':'sample_idx',
        'estimated_y2_dim_0':'target_2_dim_idx',
        'estimated_y2_dim_1':'sample_idx',
        'Sigma_11_dim_0':'target_1_dim_idx',
        'Sigma_11_dim_1':'target_1_dim_idx_bis',
        'Sigma_22_dim_0':'target_2_dim_idx',
        'Sigma_22_dim_1':'target_2_dim_idx_bis',
        'Sigma_12_dim_0':'target_1_dim_idx',
        'Sigma_12_dim_1':'target_2_dim_idx',
        'Sigma_21_dim_0':'target_2_dim_idx',
        'Sigma_21_dim_1':'target_1_dim_idx',
        'y1_predictive_dim_0':'target_1_dim_idx',
        'y1_predictive_dim_1':'sample_idx',
        'y2_predictive_dim_0':'target_2_dim_idx',
        'y2_predictive_dim_1':'sample_idx',
        'y_dim_0':'sample_idx',
        'y1_dim_0':'target_1_dim_idx',
        'y1_dim_1':'sample_idx',
        'y2_dim_0':'target_2_dim_idx',
        'y2_dim_1':'sample_idx',
        'X_dim_0':'covariate_idx',
        'X_dim_1':'sample_idx',
        'B1_dim_0':'target_1_dim_idx',
        'B1_dim_1':'basis_fun_branch_1_idx',
        'B2_dim_0':'target_2_dim_idx',
        'B2_dim_1':'basis_fun_branch_2_idx',
        'N_dim_0':'data_dim_0',
        'L1_dim_0':'data_dim_0',
        'L2_dim_0':'data_dim_0',
        'p1_dim_0':'data_dim_0',
        'p2_dim_0':'data_dim_0',
        'k_dim_0':'data_dim_0',
        'r_dim_0':'data_dim_0',
        'v_dim_0':'data_dim_0',
        'nu_dim_0':'data_dim_0',
        'alpha_1_dim_0':'data_dim_0',
        'alpha_2_dim_0':'data_dim_0',
        'alpha_psi_dim_0':'data_dim_0',
        'beta_psi_dim_0':'data_dim_0',
        'alpha_sigma_dim_0':'data_dim_0',
        'beta_sigma_dim_0':'data_dim_0',
        'psi_noise_scale_multiplier_1_dim_0':'data_dim_0',
        'theta_noise_scale_multiplier_1_dim_0':'data_dim_0',
        'psi_noise_scale_multiplier_2_dim_0':'data_dim_0',
        'theta_noise_scale_multiplier_2_dim_0':'data_dim_0',
        'log_lik_y_dim_0':'sample_idx',
        'X_test_dim_0':'covariate_idx',
        'X_test_dim_1':'sample_idx'
    }

    if isinstance(xr_datum, _xr.DataArray):
        return(
            xr_datum.rename(
                {nome:rename_dict[nome] for nome in [nome for nome in list(xr_datum.dims) if nome in list(rename_dict.keys())]}
            )        
        )
    else:
        return(
            xr_datum.rename_dims(
                {nome:rename_dict[nome] for nome in [nome for nome in list(xr_datum.dims) if nome in list(rename_dict.keys())]}
            )        
        )




def default_basis_dictionary_builder(rng_seed,
                                     N = 50,
                                     r = 5,
                                     L1 = 30,
                                     domain_range_1 = [0,1],
                                     p1 = 10,
                                     L2 = 40,
                                     domain_range_2 = [0,5],
                                     p2 = 12,
                                     k = 4,
                                     X_scale_multiplier = 0.3,
                                     nu = 5,
                                     v = 30,
                                     alpha_1 = 2.1,
                                     alpha_2 = 2.5,
                                     alpha_psi = 4.0,
                                     beta_psi = 3.0,
                                     alpha_sigma = 4.0,
                                     beta_sigma = 3.0,
                                     psi_noise_scale_multiplier_1 = 1.0,
                                     psi_noise_scale_multiplier_2 = 1.0,
                                     theta_noise_scale_multiplier_1 = 1.0,
                                     theta_noise_scale_multiplier_2 = 1.0
                                     ):
    """
    Build a data dictionary for the multitarget latent factor model with customizable parameters.

    Parameters
    ----------
    rng_seed : int
        Seed for the random number generator to ensure reproducibility.
    N : int
        Number of samples in the training set.
    r : int
        Number of covariates used in regression.
    L1 : int
        Number of positions in which the first target has been sampled.
    domain_range_1 : list
        Domain range [min, max] for the first target.
    p1 : int
        Number of basis functions used to describe the first target.
    L2 : int
        Number of positions in which the second target has been sampled.
    domain_range_2 : list
        Domain range [min, max] for the second target.
    p2 : int
        Number of basis functions used to describe the second target.
    k : int
        Number of latent factors to be fitted.
    X_scale_multiplier : float
        Multiplier for scaling the covariate matrix X.
    nu : float
        Degrees of freedom for the prior of the Regression Coefficients (β).
    v : float
        Degrees of freedom for the prior of the Latent Factors (Λ).
    alpha_1 : float
        Shape parameter for (τ_λ)_1.
    alpha_2 : float
        Shape parameter for δ.
    alpha_psi : float
        Shape parameter for ψ.
    beta_psi : float
        Rate parameter for ψ.
    alpha_sigma : float
        Shape parameter for τ_θ.
    beta_sigma : float
        Rate parameter for τ_θ.
    psi_noise_scale_multiplier_1 : float
        Multiplier for ψ noise scale.
    theta_noise_scale_multiplier_1 : float
        Multiplier for τ_θ noise scale.
    psi_noise_scale_multiplier_2 : float
        Multiplier for ψ noise scale.
    theta_noise_scale_multiplier_2 : float
        Multiplier for τ_θ noise scale.

    Raises
    ------
    ValueError
	If there is some mistake in the passed parameters

    Returns
    -------
    dict
        A dictionary containing all the data/hyperparameters for the multitarget latent factor model.

    """

    # Validate input parameters
    if N <= 0 or r <= 0 or L1 <= 0 or L2 <= 0 or p1 <= 0 or p2 <= 0 or k <= 0:
        raise ValueError("N, r, L1, L2, p1, p2, and k must be positive integers.")
    if (not isinstance(N , (int, _np.int_))) or (not isinstance(r, (int, _np.int_))) or (not isinstance(L1, (int, _np.int_))) or (not isinstance(L2, (int, _np.int_))) or (not isinstance(p1, (int, _np.int_))) or (not isinstance(p2, (int, _np.int_))) or (not isinstance(k, (int, _np.int_))):
        raise ValueError("N, r, L1, L2, p1, p2, and k must be positive integers.")
    if domain_range_1[0] >= domain_range_1[1] or domain_range_2[0] >= domain_range_2[1]:
        raise ValueError("Domain range must have a lower bound less than the upper bound.")
    if not isinstance(domain_range_1, list) or not isinstance(domain_range_2, list) or len(domain_range_1) != 2 or len(domain_range_2) != 2:
        raise ValueError("Domain ranges must be lists of two elements.")
    if nu <= 0 or v <= 0 or alpha_1 <= 0 or alpha_2 <= 0 or alpha_psi <= 0 or beta_psi <= 0 or alpha_sigma <= 0 or beta_sigma <= 0:
        raise ValueError("nu, v, alpha_1, alpha_2, alpha_psi, beta_psi, alpha_sigma, and beta_sigma must be positive.")
    if not isinstance(rng_seed, (int, _np.int_)):
        raise ValueError("rng_seed must be an integer")    

    # Initialize the random number generator
    rng = _np.random.default_rng(rng_seed)
    
    # The vector of sampling locations
    t1 = _np.linspace(domain_range_1[0], domain_range_1[1], L1)
    t2 = _np.linspace(domain_range_2[0], domain_range_2[1], L2)

    
    # The basis matrices are made up of B-Splines, plus a constant column (like in the original model)
    Bspline1 = \
    scikit_basis_funs.BSplineBasis(
        domain_range=domain_range_1,
        n_basis=p1-1,
        order=3
    )
    B1 = \
    _np.concatenate(
        [
            _np.ones((L1,1)),
            Bspline1(t1)[:,:,0].T
        ],
        axis=1
    )
    Bspline2 = \
    scikit_basis_funs.BSplineBasis(
        domain_range=domain_range_2,
        n_basis=p2-1,
        order=3
    )
    B2 = \
    _np.concatenate(
        [
            _np.ones((L2,1)),
            Bspline2(t2)[:,:,0].T
        ],
        axis=1
    )
    
    data_dic = {
        'N': N, 'L1': L1, 'L2': L2, 'p1': p1, 'p2': p2, 'k': k, 'r': r,
        'y1': _np.zeros((L1, N)), 'y2': _np.zeros((L2, N)),
        'X': X_scale_multiplier * rng.normal(size=(r, N)),
        'B1': B1, 'B2': B2,
        'nu':nu, 'v': v,
        'alpha_1': alpha_1, 'alpha_2': alpha_2,
        't1': t1, 't2': t2,
        'alpha_psi': alpha_psi, 'beta_psi': beta_psi,
        'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
        'psi_noise_scale_multiplier_1': psi_noise_scale_multiplier_1,
        'psi_noise_scale_multiplier_2': psi_noise_scale_multiplier_2,
        'theta_noise_scale_multiplier_1': theta_noise_scale_multiplier_1,
        'theta_noise_scale_multiplier_2': theta_noise_scale_multiplier_2
    }

    return data_dic


    
def compute_BLambaBetaX(B_xr, Λ_xr, β_xr, X_xr,
                          B_dims = ['target_1_dim_idx','basis_fun_branch_1_idx'],
                          Λ_dims = ['basis_fun_branch_1_idx','latent_factor_idx'],
                          β_dims = ['latent_factor_idx','covariate_idx'],
                          X_dims = ['covariate_idx','sample_idx']
                          ):
    """
    Build a data dictionary for the multitarget latent factor model with customizable parameters.

    Parameters
    ----------
    B_xr : xr.DataArray
	An xarray DataArray containing the basis matrix
    Λ_xr : xr.DataArray
	An xarray DataArray containing the latent factors (Λ)
    β_xr : xr.DataArray
	An xarray DataArray containing the regression coefficients (β)
    X_xr : xr.DataArray
	An xarray DataArray containing the covariates
    B_dims : list
	A list containing the 2 dimensions of B relevant for the necessary matrix multiplication
    Λ_dims : list
	A list containing the 2 dimensions of Λ relevant for the necessary matrix multiplication
    β_dims : list
	A list containing the 2 dimensions of β relevant for the necessary matrix multiplication
    X_dims : list
	A list containing the 2 dimensions of X relevant for the necessary matrix multiplication

    Raise
    -----
    ValueError
	If some parameter doesn't respect the rules expressed in the Parameters block of this docstring

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the expected value of Y given the covariates

    """

    # Check if inputs are instances of xr.DataArray
    if not all(isinstance(arr, _xr.DataArray) for arr in [B_xr, Λ_xr, β_xr, X_xr]):
        raise ValueError("All input arrays must be instances of xr.DataArray")
    
    # Check if dimensions are valid
    for dims, arr_name in zip([B_dims, Λ_dims, β_dims, X_dims], ['B_xr', 'Λ_xr', 'β_xr', 'X_xr']):
        if not all(dim in arr.dims for dim in dims for arr in [B_xr, Λ_xr, β_xr, X_xr]):
            raise ValueError(f"One or more dimensions specified for {arr_name} are invalid")
    
    # Check if dimensions are consistent for matrix multiplication
    if len(B_dims) != 2 or len(Λ_dims) != 2 or len(β_dims) != 2 or len(X_dims) != 2:
        raise ValueError("Dimensions lists must contain exactly 2 dimensions for matrix multiplication")
    
    return(
        _xrein.linalg.matmul(
            B_xr,
            _xrein.linalg.matmul(
                Λ_xr,
                _xrein.linalg.matmul(
                    β_xr,
                    X_xr,
                    dims=[[β_dims[0],β_dims[1]],[X_dims[0], X_dims[1]]]
                ),
                dims=[[Λ_dims[0], Λ_dims[1]],[β_dims[0], X_dims[1]]]
            ),
            dims=[[B_dims[0], B_dims[1]],[Λ_dims[0],X_dims[1]]]
        )
    )





def sample_from_posterior(data_dic,
                        rng_seed,
                        stan_file_path=None,
                        output_dir='./out',
                        laplace_draws = 100,
                        iter_warmup = 500,
                        iter_sampling = 1000,
                        prior_draws = 1000,
                        max_treedepth = 12,
                        X_test = None,
                        ):
    """
    Executes the Hamiltonian Monte Carlo (HMC) sampling for the multitarget latent factor model,
    incorporating Laplace approximation for initialization and generating prior and posterior samples.
    If a test set is provided, it will also compute the predictive prior and posterior
    samples for that test set.
    The function organizes output directories, prepares initial conditions, and
    conducts HMC sampling using a specified Stan model.

    Parameters
    ----------
    data_dic : dict
        A dictionary containing all the data and hyperparameters required for the model.
        
    rng_seed : int
        Seed for the random number generator to ensure reproducibility.
        
    stan_file : str, optional
        Path to the Stan model file.
        
    output_dir : str, optional
        Base directory to store output files. Defaults to './out'.
        
    laplace_draws : int, optional
        Number of draws to sample using Laplace approximation. Defaults to 100.
        
    iter_warmup : int, optional
        Number of warmup iterations for HMC. Defaults to 500.
        
    iter_sampling : int, optional
        Number of sampling iterations for HMC. Defaults to 1000.
        
    prior_draws : int, optional
        Number of draws to sample from the prior. Defaults to 1000.
        
    max_treedepth : int, optional
        XD. Defaults to 12.
        
    X_test : np.ndarray, optional
        A 2D numpy array of test set covariates. If provided, the function will also generate prior and posterior samples for this test set.

    Returns
    -------
    arviz.InferenceData
        An ArviZ InferenceData object containing the results of the HMC sampling, prior samples, and, if X_test is provided, predictions for the test set.

    Notes
    -----
    The function performs several steps:
    1. Preparation of output directories.
    2. Laplace approximation to find mode of the posterior and to sample around it.
    3. Initialization of HMC sampler based on Laplace samples.
    4. Execution of HMC sampling.
    5. Sampling from the prior using the `sample_from_prior` function.
    6. If X_test compute the predictives for that set.
    7. Assembly and return of all sampling results in an ArviZ InferenceData object.
    
    As a side effect, this function will alter the directory passed through output_dir, creating one if it's missing.

    References
    ----------
    Mention any references to the theoretical background, Stan documentation, or relevant literature.

    See Also
    --------
    sample_from_prior: For sampling from the prior.
    sample_from_posterior_predictive: For generating posterior predictive samples.
    """
    
    # Organize the directory to save the results
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

    try:
        _os.mkdir(output_dir + '/MAP_Laplace_outDir')
    except Exception as e:
        print('While creating the output directory for the Laplace Sampler')
        print(e)
        pass

    if stan_file_path == None:
        stan_file = pkg_resources.resource_filename(__name__, 'model.stan')
    else:
        stan_file = stan_file_path

    # Initialize random number generator
    rng = _np.random.default_rng(rng_seed)

    # Laplace Sample about the mode
    idata_MAP = \
    MAP_and_LaplaceSample(
        data_dic,
        rng.integers(1000000),
        stan_file=stan_file,
        output_dir= (output_dir + '/MAP_Laplace_outDir'),
        lista_observed=['y1','y2'],
        lista_lklhood=['y'],
        lista_predictive=['y1','y2'],
        laplace_draws=laplace_draws,
    )

    # Use the Laplace Samples to build the initial conditions of the Hamiltonian Monte Carlo Sampler
    aux_dict = idata_MAP['Laplace_inference_data'].posterior.to_dict()    
    inits_dict = []
    for i in range(4):
        init_dict = {}
        for key in ['theta1', 'theta2', 'Lambda1', 'Lambda2', 'eta', 'beta', 'tau_theta1', 'tau_theta2', 'delta1', 'delta2', 'psi1', 'psi2']:
            if key in ['psi1', 'psi2']:
                init_dict[key] = idata_MAP['Laplace_inference_data'].posterior[key][0, i].values
            elif key in ['delta1','delta2','tau_theta1','tau_theta2']:
                init_dict[key] = idata_MAP['Laplace_inference_data'].posterior[key][0, i, :].values
            else:
                init_dict[key] = idata_MAP['Laplace_inference_data'].posterior[key][0, i, :, :].values
        inits_dict.append(init_dict)

    # Hamiltonian Monte Carlo
    idata = \
    _execute_HMC(
        data_dic,
        rng.integers(1000000),
        stan_file=stan_file,
        output_dir=(output_dir + '/HMC_outDir'),
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        max_treedepth=max_treedepth,
        inits=inits_dict,
        lista_observed=['y1','y2'],
        lista_lklhood=['y'],
        lista_predictive=['y1','y2']
    )

    # Sampling from the Prior
    if X_test is not None:
        prior_samples_xr = sample_from_prior(
            data_dic,
            X_test,
            rng.integers(1000000),
            prior_draws
        )
    else:
        prior_samples_xr = sample_from_prior(
            data_dic,
            rng.normal(size=(data_dic['r'], 2)),
            rng.integers(1000000),
            prior_draws
        )

    # Add the prior samples inside the inference data object and fix some naming of variables
    idata2 = \
    idata.rename(
        name_dict={
            'y1_predictive':'y1',
            'y2_predictive':'y2',        
        },
        groups='posterior_predictive'
    ).rename(
        name_dict={
            'y': 'y_posterior'
        },
        groups='log_likelihood'
    )
    idata2.add_groups(
        {'prior': prior_samples_xr.drop(['log_lik_y', 'y1_predictive', 'y2_predictive', 'y1_test_predictive', 'y2_test_predictive'])}
    )
    idata2.add_groups(
        {'prior_predictive': prior_samples_xr[['y1_predictive','y2_predictive']]}
    )
    idata = idata2.assign(y_prior=prior_samples_xr['log_lik_y'].rename({'log_lik_y_dim_0':'y_dim_0'}), groups='log_likelihood')

    # Add the prior predictions on X_test
    if X_test is not None:
        idata.add_groups(
            {'predictions': prior_samples_xr[['y1_test_predictive','y2_test_predictive']].rename({'y1_test_predictive':'y1_test_prior_predictive', 'y2_test_predictive':'y2_test_prior_predictive'})}
        )
        idata.add_groups(
            {
                'predictions_constant_data': _xr.Dataset(
                    {
                        'N': X_test.shape[1],
                        'X': (('X_dim_0','X_dim_1'), X_test)
                    }
                )
            }
        )
        aux_xr = sample_from_posterior_predictive(rng.integers(1000000), idata, X_test)
        idata = idata = idata.assign(
            y1_test_posterior_predictive = aux_xr['y1_posterior_predictive'],
            y2_test_posterior_predictive = aux_xr['y2_posterior_predictive'],
            groups = 'predictions'
        )
        
    
    return(idata)


    
def sample_from_posterior_predictive(rng_seed, idata, X_test):
    """
    Samples from the posterior predictive distribution of a multitarget latent factor model for a given test set X_test.
    This function utilizes the posterior samples of model parameters stored in an ArviZ InferenceData object to generate
    predictive samples for new data.

    Parameters
    ----------
    rng_seed : int
    	Seed for the random number generator to ensure reproducibility.
    
    idata : arviz.InferenceData
        An ArviZ InferenceData object containing the posterior samples of the model parameters.
        
    X_test : np.ndarray
        A 2D numpy array of the covariates for the test set. The array should have a shape of (r, N_test),
        where r is the number of covariates and N_test is the number of test samples.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the posterior predictive samples for the test set. The Dataset includes two main variables:
        'y1_posterior_predictive' and 'y2_posterior_predictive', corresponding to the predictive samples for the first and second
        target variables, respectively.

    Notes
    -----
    This function calculates the expected values of the target variables for the test set using the regression coefficients and
    the covariate values in X_test. It then samples from the multivariate normal distribution defined by these expected values
    and the covariance structure estimated from the posterior samples.

    References
    ----------
    For theoretical background and model specifics, refer to relevant literature on Bayesian hierarchical modeling and
    the specific multitarget latent factor model implementation.

    See Also
    --------
    execute_HMC: For executing HMC sampling to obtain posterior samples.
    sample_from_prior: For sampling from the model's prior distribution.
    """
    
    rng = _np.random.default_rng(rng_seed)

    N = X_test.shape[1]
    L1 = idata.constant_data['L1'].values[0]
    L2 = idata.constant_data['L2'].values[0]

    estimated_y1_xr = \
    _xrein.linalg.matmul(
        renaming_convention( idata.posterior['regr_coeffs1'] ),
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['target_1_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    )
    estimated_y2_xr = \
    _xrein.linalg.matmul(
        renaming_convention( idata.posterior['regr_coeffs2'] ),
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['target_2_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    )

    y1_predictive = _xr.DataArray(_np.zeros((estimated_y1_xr.sizes['chain'],estimated_y1_xr.sizes['draw'],L1,N)), dims=['chain','draw','target_1_dim_idx','sample_idx'])
    y2_predictive = _xr.DataArray(_np.zeros((estimated_y1_xr.sizes['chain'],estimated_y1_xr.sizes['draw'],L2,N)), dims=['chain','draw','target_2_dim_idx','sample_idx'])
    
    for c in range(estimated_y1_xr.sizes['chain']):
        for d in range(estimated_y1_xr.sizes['draw']):
            Σ11 = idata.posterior['Sigma_11'].sel(chain = c , draw = d).values
            Σ22 = idata.posterior['Sigma_22'].sel(chain = c , draw = d).values
            Σ12 = idata.posterior['Sigma_12'].sel(chain = c , draw = d).values
            
            Σ = assemble_Σ(Σ11, Σ22, Σ12)
            cov_obj_Σ = _spst.Covariance.from_cholesky( _np.linalg.cholesky(Σ) )

            # Compute log likelihoods and prior predictives
            y_predictive = _np.zeros((L1+L2, N))

            for i in range(N):
                estimated_y_aux = _np.zeros(L1+L2)
                estimated_y_aux[0:L1] = estimated_y1_xr.sel(chain = c, draw = d, sample_idx = i).values
                estimated_y_aux[L1:(L1+L2)] = estimated_y2_xr.sel(chain = c, draw = d, sample_idx = i).values

                MVN = _spst.multivariate_normal( mean = estimated_y_aux, cov = cov_obj_Σ )

                y_predictive_aux = MVN.rvs(random_state = rng)[0]

                y1_predictive[c,d,:,i] = y_predictive_aux[0:L1]
                y2_predictive[c,d,:,i] = y_predictive_aux[L1:(L1+L2)]
    
    return(
        _xr.Dataset(
            {
                'y1_posterior_predictive': y1_predictive,
                'y2_posterior_predictive': y2_predictive
            }
        )
    )


def dataset_generator(rng_seed,
                      n_train_samples = 50,
                      n_test_samples = 20,
                      r = 5,
                      L1 = 30,
                      domain_range_1 = [0,1],
                      p1 = 10,
                      L2 = 40,
                      domain_range_2 = [0,5],
                      p2 = 12,
                      k = 4,
                      X_scale_multiplier = 0.3,
                      nu = 5,
                      v = 30,
                      alpha_1 = 2.1,
                      alpha_2 = 2.5,
                      alpha_psi = 4.0,
                      beta_psi = 3.0,
                      alpha_sigma = 4.0,
                      beta_sigma = 3.0,
                      psi_noise_scale_multiplier_1 = 1.0,
                      psi_noise_scale_multiplier_2 = 1.0,
                      theta_noise_scale_multiplier_1 = 1.0,
                      theta_noise_scale_multiplier_2 = 1.0
                      ):
    """
    Generates a synthetic dataset for a multitarget latent factor model, including training and test sets, based on specified
    parameters and hyperparameters. This function is useful for simulation studies and model testing.

    Parameters
    ----------
    rng_seed : int
        Seed for the random number generator to ensure reproducibility.
        
    n_train_samples : int, optional
        Number of training samples to generate. Defaults to 50.
        
    n_test_samples : int, optional
        Number of test samples to generate. Defaults to 20.
        
    r : int, optional
        Number of covariates used in the regression. Defaults to 5.
        
    L1, L2 : int, optional
        Number of positions at which observations for the first and second targets are made, respectively.
        
    domain_range_1, domain_range_2 : list, optional
        The range of the domain for the first and second targets, respectively.
        
    p1, p2 : int, optional
        Number of basis functions used to describe the first and second targets, respectively.
        
    k : int, optional
        Number of latent factors in the model.
        
    X_scale_multiplier : float, optional
        Multiplier to scale the covariates' values. Defaults to 0.3.
        
    nu, v : int, optional
        Degrees of freedom for the priors of regression coefficients and latent factors, respectively.
        
    alpha_1, alpha_2, alpha_psi, alpha_sigma : float, optional
        Shape parameters for the gamma distributions of various model components.
        
    beta_psi, beta_sigma : float, optional
        Rate parameters for the gamma distributions of psi and sigma components.
        
    psi_noise_scale_multiplier_1, psi_noise_scale_multiplier_2, theta_noise_scale_multiplier_1, theta_noise_scale_multiplier_2 : float, optional
        Scale multipliers for noise terms in the model.

    Returns
    -------
    tuple
        A tuple containing two elements:
        1. A dictionary with the generated data and hyperparameters for the training set.
        2. An xarray Dataset with the generated data for the test set and additional simulation details.

    Notes
    -----
    This function first generates basis functions and associated parameters according to the specified model settings.
    It then samples from the prior distribution of the model to create synthetic observations for both training and test sets.

    See Also
    --------
    sample_from_prior: Used internally to generate samples from the model's prior.
    execute_HMC: For executing HMC sampling with the generated dataset.
    """
    
    rng = _np.random.default_rng(rng_seed)
    data_dic = default_basis_dictionary_builder(
        rng.integers(1000000),
        N = n_train_samples + n_test_samples
    )
    idata_simul = \
    sample_from_prior(
        data_dic, rng.normal(size=(data_dic['r'],2)), rng.integers(1000000), 10
    )
    data_dic['y1'] = idata_simul['y1_predictive'].sel(sample=0)[:,0:n_train_samples].values
    data_dic['y2'] = idata_simul['y2_predictive'].sel(sample=0)[:,0:n_train_samples].values
    
    y1_test = idata_simul['y1_predictive'].sel(sample=0)[:,n_train_samples:]
    y2_test = idata_simul['y2_predictive'].sel(sample=0)[:,n_train_samples:]
    
    X_test = data_dic['X'][:,n_train_samples:]

    data_dic['X'] = data_dic['X'][:,0:n_train_samples]
    data_dic['N'] = n_train_samples

    aux_xr = \
    _xr.Dataset(
        {var: idata_simul[var].sel(sample=0) for var in idata_simul.data_vars}
    ).drop(['log_lik_y','y1_predictive','y2_predictive','y1_test_predictive','y2_test_predictive'])

    aux_xr['X_test'] = _xr.DataArray(X_test, dims=['X_test_dim_0','X_test_dim_1'])
    aux_xr['y1'] = y1_test
    aux_xr['y2'] = y2_test
    
    return data_dic, aux_xr


def one_functional_target_dictionary_builder(rng_seed,
                                             N = 50,
                                             r = 5,
                                             L1 = 3,
                                             L2 = 40,
                                             domain_range_2 = [0,5],
                                             p2 = 12,
                                             k = 4,
                                             X_scale_multiplier = 0.3,
                                             nu = 5,
                                             v = 30,
                                             alpha_1 = 2.1,
                                             alpha_2 = 2.5,
                                             alpha_psi = 4.0,
                                             beta_psi = 3.0,
                                             alpha_sigma = 4.0,
                                             beta_sigma = 3.0,
                                             psi_noise_scale_multiplier_1 = 1.0,
                                             psi_noise_scale_multiplier_2 = 1.0,
                                             theta_noise_scale_multiplier_1 = 1.0,
                                             theta_noise_scale_multiplier_2 = 1.0
                                             ):
    """
    Build a data dictionary for the multitarget latent factor model with customizable parameters,
    in the case of one functional target and a multivariate one.

    Parameters
    ----------
    rng_seed : int
        Seed for the random number generator to ensure reproducibility.
    N : int
        Number of samples in the training set.
    r : int
        Number of covariates used in regression.
    L1 : int
        Dimensionality of the multivariate target.
    L2 : int
        Number of positions in which the functional target has been sampled.
    domain_range_2 : list
        Domain range [min, max] for the functional target.
    p2 : int
        Number of basis functions used to describe the functional target.
    k : int
        Number of latent factors to be fitted.
    X_scale_multiplier : float
        Multiplier for scaling the covariate matrix X.
    nu : float
        Degrees of freedom for the prior of the Regression Coefficients (β).
    v : float
        Degrees of freedom for the prior of the Latent Factors (Λ).
    alpha_1 : float
        Shape parameter for (τ_λ)_1.
    alpha_2 : float
        Shape parameter for δ.
    alpha_psi : float
        Shape parameter for ψ.
    beta_psi : float
        Rate parameter for ψ.
    alpha_sigma : float
        Shape parameter for τ_θ.
    beta_sigma : float
        Rate parameter for τ_θ.
    psi_noise_scale_multiplier_1 : float
        Multiplier for ψ noise scale.
    theta_noise_scale_multiplier_1 : float
        Multiplier for τ_θ noise scale.
    psi_noise_scale_multiplier_2 : float
        Multiplier for ψ noise scale.
    theta_noise_scale_multiplier_2 : float
        Multiplier for τ_θ noise scale.

    Raises
    ------
    ValueError
	If there is some mistake in the passed parameters

    Returns
    -------
    dict
        A dictionary containing all the data/hyperparameters for the multitarget latent factor model.

    """

    # Validate input parameters
    if N <= 0 or r <= 0 or L1 <= 0 or L2 <= 0 or  p2 <= 0 or k <= 0:
        raise ValueError("N, r, L1, L2, p2, and k must be positive integers.")
    if (not isinstance(N , (int, _np.int_))) or (not isinstance(r, (int, _np.int_))) or (not isinstance(L1, (int, _np.int_))) or (not isinstance(L2, (int, _np.int_))) or (not isinstance(p2, (int, _np.int_))) or (not isinstance(k, (int, _np.int_))):
        raise ValueError("N, r, L1, L2, p2, and k must be positive integers.")
    if domain_range_2[0] >= domain_range_2[1]:
        raise ValueError("Domain range must have a lower bound less than the upper bound.")
    if not isinstance(domain_range_2, list) or len(domain_range_2) != 2:
        raise ValueError("Domain ranges must be lists of two elements.")
    if nu <= 0 or v <= 0 or alpha_1 <= 0 or alpha_2 <= 0 or alpha_psi <= 0 or beta_psi <= 0 or alpha_sigma <= 0 or beta_sigma <= 0:
        raise ValueError("nu, v, alpha_1, alpha_2, alpha_psi, beta_psi, alpha_sigma, and beta_sigma must be positive.")
    if not isinstance(rng_seed, (int, _np.int_)):
        raise ValueError("rng_seed must be an integer")    

    # Initialize the random number generator
    rng = _np.random.default_rng(rng_seed)
    
    # The vector of sampling locations
    t1 = _np.arange(L1)
    t2 = _np.linspace(domain_range_2[0], domain_range_2[1], L2)

    B1 = _np.eye(L1,L1)
    Bspline2 = \
    scikit_basis_funs.BSplineBasis(
        domain_range=domain_range_2,
        n_basis=p2-1,
        order=3
    )
    B2 = \
    _np.concatenate(
        [
            _np.ones((L2,1)),
            Bspline2(t2)[:,:,0].T
        ],
        axis=1
    )
    
    data_dic = {
        'N': N, 'L1': L1, 'L2': L2, 'p1': L1, 'p2': p2, 'k': k, 'r': r,
        'y1': _np.zeros((L1, N)), 'y2': _np.zeros((L2, N)),
        'X': X_scale_multiplier * rng.normal(size=(r, N)),
        'B1': B1, 'B2': B2,
        'nu':nu, 'v': v,
        'alpha_1': alpha_1, 'alpha_2': alpha_2,
        't1': t1, 't2': t2,
        'alpha_psi': alpha_psi, 'beta_psi': beta_psi,
        'alpha_sigma': alpha_sigma, 'beta_sigma': beta_sigma,
        'psi_noise_scale_multiplier_1': psi_noise_scale_multiplier_1,
        'psi_noise_scale_multiplier_2': psi_noise_scale_multiplier_2,
        'theta_noise_scale_multiplier_1': theta_noise_scale_multiplier_1,
        'theta_noise_scale_multiplier_2': theta_noise_scale_multiplier_2
    }

    return data_dic


def sample_conditional_predictive(idata, X_test, rng_seed, group = "posterior", Y1_test = None, Y2_test = None, bootstrap = None, required = "predictive"):
    """
    Samples from the conditional predictive distribution of a multitarget latent factor model.

    Depending on the specified parameters, this function can sample from the conditional predictive distributions
    given the posterior or prior information. It supports various forms of predictive distributions, including
    ones incorporating idiosyncratic variations and noise components. The function can also handle bootstrapping
    to enable downsampling of the Monte Carlo samples for computational efficiency.

    Parameters
    ----------
    idata : az.InferenceData
        An ArviZ InferenceData structure containing the posterior/prior distributions. This dataset should contain
        relevant model parameters and statistics for either the posterior or prior group.
        
    X_test : np.ndarray
        A 2D numpy array of floats representing the covariates for the test dataset. The dimensions should match
        the requirements for the covariate space of the model.

    rng_seed : int
        Seed for the random number generator to ensure reproducible results.

    group : str, optional
        Specifies which group of distributions to sample from. Can be either 'posterior' or 'prior'. The default is 'posterior'.

    Y1_test : np.ndarray, optional
        A 2D numpy array representing the observed values for the first target variable in the test set.
        If provided, the function will sample conditional predictive distributions for the other target variable.
        Cannot be used simultaneously with Y2_test.

    Y2_test : np.ndarray, optional
        A 2D numpy array representing the observed values for the second target variable in the test set.
        If provided, the function will sample conditional predictive distributions for the other target variable.
        Cannot be used simultaneously with Y1_test.

    bootstrap : int, optional
        If specified, this integer value is used to resample from the prior/posterior samples to enable
        downsampling for faster computation. If None, the full prior/posterior samples are used.

    required : str, optional
        Specifies the type of predictive distribution to sample. Options are 'predictive' (default), 
        'predictive idiosyncratic', or 'predictive estimate'. Each option dictates the level of detail
        and the inclusion of noise components in the predictive distributions.

    Raises
    ------
    ValueError
        If both Y1_test and Y2_test are provided simultaneously or both are missing.
        If an unsupported value for `required` is provided.
        If an unsupported value for `group` is provided.
        If there is an incompatiblity in any of the dimensions.
        If idata is missing the "constant_data" section

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the sampled conditional predictive distributions. The structure of the
        array depends on the selected group, whether bootstrap resampling is used, and the type of predictive
        distribution requested.

    Notes
    -----
    This function is designed to work with models where the relationship between covariates and target variables
    can be described through a latent factor model. It allows for flexible sampling strategies, including conditional
    sampling based on observed data for one of the target variables and adjusting the complexity of the sampled
    distributions through various 'required' settings.

    The conditional predictive sampling process takes into account the correlations between target variables, leveraging
    the covariance structures present in the model's formulation. This approach enables nuanced predictions that
    reflect the underlying model dynamics and observed data characteristics.

    TO-DO Clarify this note

    References
    ----------
    If the function is based on published research or algorithms, cite these sources here.

    See Also
    --------
    Mention related functions or classes here. For example, functions for sampling unconditional predictive distributions
    or functions for parameter estimation within the same model framework.

    """

    from copy import deepcopy

    # Input type and value checks
    if not isinstance(idata, _az.InferenceData):
        raise TypeError("idata must be an arviz inference data.")

    if not (isinstance(X_test, _np.ndarray) and X_test.ndim == 2):
        raise TypeError("X_test must be a 2D numpy array.")

    if not isinstance(rng_seed, (int, _np.int_)):
        raise TypeError("rng_seed must be an integer.")

    if group not in ['posterior', 'prior']:
        raise ValueError("group must be either 'posterior' or 'prior'.")

    if not (Y1_test is None or (isinstance(Y1_test, _np.ndarray) and Y1_test.ndim == 2)):
        raise TypeError("Y1_test must be a 2D numpy array or None.")

    if not (Y2_test is None or (isinstance(Y2_test, _np.ndarray) and Y2_test.ndim == 2)):
        raise TypeError("Y2_test must be a 2D numpy array or None.")

    if Y1_test is not None and Y2_test is not None:
        raise ValueError("Y1_test and Y2_test cannot both be provided.")

    if bootstrap is not None and not isinstance(bootstrap, (int, _np.int_)):
        raise TypeError("bootstrap must be an integer or None.")

    if required not in ['predictive', 'predictive idiosyncratic', 'predictive estimate']:
        raise ValueError("required must be one of 'predictive', 'predictive idiosyncratic', or 'predictive estimate'.")

    # Compatibility checks
    if 'constant_data' in idata.groups():
        if X_test.shape[0] != idata.constant_data['r'].values[0]:
            raise ValueError(f"X_test.shape[0] must match the declared number of covariates")
        
        if Y1_test is not None:
            if Y1_test.shape[0] != idata.constant_data['L1'].values[0]:
                raise ValueError("Y1_test.shape[0] must match the declared number of sampling locations for target 1")
            if Y1_test.shape[1] != X_test.shape[1]:
                raise ValueError("Y1_test.shape[1] must match X_test.shape[1]")
        
        if Y2_test is not None:
            if Y2_test.shape[0] != idata.constant_data['L2'].values[0]:
                raise ValueError("Y2_test shape[0] must match the declared number of sampling locations for target 2")
            if Y2_test.shape[1] != X_test.shape[1]:
                raise ValueError("Y2_test.shape[1] must match X_test.shape[1]")                
    else:
        raise ValueError("idata does not contain 'constant_data', required for shape compatibility checks.")


    rng = _np.random.default_rng(rng_seed)

    if group == "posterior":
        xr_dataset = deepcopy( idata.posterior.stack(sample = ('chain','draw')) )
    elif group == "prior":
        xr_dataset = deepcopy( idata.prior )
    else:
        raise ValueError("group must be one of 'prior' or 'posterior'")

    if not bootstrap is None:
        xr_dataset = xr_dataset.isel(sample=rng.integers(xr_dataset.sizes['sample'], size=bootstrap))

    if not Y1_test is None:
        if not Y2_test is None:
            raise ValueError("Y1_test and Y2_test can't both be known")
        Y_test = Y1_test
        regr_coeffs = renaming_convention( xr_dataset['regr_coeffs2'] )
        regr_coeffs_in = renaming_convention( xr_dataset['regr_coeffs1'] )
        unknown_target = 2
        known_target = 1
    elif not Y2_test is None:
        Y_test = Y2_test
        regr_coeffs = renaming_convention( xr_dataset['regr_coeffs1'] )
        regr_coeffs_in = renaming_convention( xr_dataset['regr_coeffs2'] )
        unknown_target = 1
        known_target = 2
    else:
        raise ValueError("Either Y1_test or Y2_test has to be known, did you mean to call sample_unconditional_predictive?")

    estimate_Y = \
    _xrein.linalg.matmul(
        regr_coeffs,
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[[f'target_{unknown_target}_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    ).rename({f'target_{unknown_target}_dim_idx':'target_out_idx'})

    estimate_Yin = \
    _xrein.linalg.matmul(
        regr_coeffs_in,
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[[f'target_{known_target}_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    ).rename({f'target_{known_target}_dim_idx':'target_in_idx'})

    Σout = renaming_convention( xr_dataset[f'Sigma_{unknown_target}{unknown_target}'] )
    Σout_in = renaming_convention( xr_dataset[f'Sigma_{unknown_target}{known_target}'] )
    Σin_out = renaming_convention( xr_dataset[f'Sigma_{known_target}{unknown_target}'] )
    Σin = renaming_convention( xr_dataset[f'Sigma_{known_target}{known_target}'] )

    Σout = Σout.rename({old:new for old, new in zip(sorted(Σout.dims), ['sample','target_out_idx','target_out_idx2'])})
    Σin = Σin.rename({old:new for old, new in zip(sorted(Σin.dims), ['sample','target_in_idx','target_in_idx2'])})

    Σout_in = Σout_in.rename({f'target_{unknown_target}_dim_idx':'target_out_idx', f'target_{known_target}_dim_idx':'target_in_idx'})
    Σin_out = Σin_out.rename({f'target_{unknown_target}_dim_idx':'target_out_idx', f'target_{known_target}_dim_idx':'target_in_idx'})

    inverseΣin = \
    _xrein.linalg.inv(
        Σin,
        dims=['target_in_idx','target_in_idx2']
    )

    estimate_Y += \
    _xrein.linalg.matmul(
        Σout_in,
        _xrein.linalg.matmul(
            inverseΣin,
            _xr.DataArray(Y_test, dims=['target_in_idx','sample_idx']) - estimate_Yin,
            dims=[['target_in_idx','target_in_idx2'],['target_in_idx','sample_idx']]
        ),
        dims=[['target_out_idx','target_in_idx'],['target_in_idx','sample_idx']]
    )

    if required == 'predictive estimate':
        # Correctly format the predictive estimate
        Y_predictive = estimate_Y
    else: 
        Σout -= \
        _xrein.linalg.matmul(
            Σout_in,
            _xrein.linalg.matmul(
                inverseΣin,
                Σin_out,
                dims=[['target_in_idx','target_in_idx2'],['target_in_idx','target_out_idx']]
            ),
            dims=[['target_out_idx','target_in_idx'],['target_in_idx','target_out_idx']]
        )

        if required == 'predictive idiosyncratic':
            Σout -= \
            _np.square( xr_dataset[f'psi{unknown_target}']*_xr.DataArray(0.99*_np.eye(Σout.sizes['target_out_idx']), dims=['target_out_idx','target_out_idx2']) )

        # Random sample to get the predictive/predictive idiosyncratic
        Y_predictive = estimate_Y + \
        _xrein.linalg.matmul(
            _xrein.linalg.cholesky(
                Σout,
                dims=['target_out_idx','target_out_idx2']
            ),
            _xr.DataArray(
                rng.normal(size=tuple( [estimate_Y.sizes[val] for val in estimate_Y.sizes] )),
                dims=[val for val in estimate_Y.sizes]
            ),
            dims=[['target_out_idx','target_out_idx2'],['target_out_idx','sample_idx']]
        )

    if (bootstrap is None) and (group == "posterior"):
        return( Y_predictive.unstack('sample').transpose('chain','draw','target_out_idx','sample_idx') )

    return(Y_predictive)


def sample_unconditional_predictive(idata, X_test, rng_seed, group = "posterior", bootstrap = None, required = "predictive"):
    """
    Samples from the unconditional predictive distribution of a multitarget latent factor model.

    Depending on the specified parameters, this function can sample from the unconditional predictive distributions
    given the posterior or prior information. It supports various forms of predictive distributions, including
    ones incorporating idiosyncratic variations and noise components. The function can also handle bootstrapping
    to enable downsampling of the Monte Carlo samples for computational efficiency.

    Parameters
    ----------
    idata : az.InferenceData
        An ArviZ InferenceData structure containing the posterior/prior distributions. This dataset should contain
        relevant model parameters and statistics for either the posterior or prior group.
        
    X_test : np.ndarray
        A 2D numpy array of floats representing the covariates for the test dataset. The dimensions should match
        the requirements for the covariate space of the model.

    rng_seed : int
        Seed for the random number generator to ensure reproducible results.

    group : str, optional
        Specifies which group of distributions to sample from. Can be either 'posterior' or 'prior'. The default is 'posterior'.

    bootstrap : int, optional
        If specified, this integer value is used to resample from the prior/posterior samples to enable
        downsampling for faster computation. If None, the full prior/posterior samples are used.

    required : str, optional
        Specifies the type of predictive distribution to sample. Options are 'predictive' (default), 
        'predictive idiosyncratic', or 'predictive estimate'. Each option dictates the level of detail
        and the inclusion of noise components in the predictive distributions.

    Raises
    ------
    ValueError
        If an unsupported value for `required` is provided.
        If an unsupported value for `group` is provided.
        If there is an incompatiblity in any of the dimensions.
        If idata is missing the "constant_data" section

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the sampled conditional predictive distributions. The structure of the
        array depends on the selected group, whether bootstrap resampling is used, and the type of predictive
        distribution requested.

    Notes
    -----
    This function is designed to work with models where the relationship between covariates and target variables
    can be described through a latent factor model. It allows for flexible sampling strategies, including conditional
    sampling based on observed data for one of the target variables and adjusting the complexity of the sampled
    distributions through various 'required' settings.

    The conditional predictive sampling process takes into account the correlations between target variables, leveraging
    the covariance structures present in the model's formulation. This approach enables nuanced predictions that
    reflect the underlying model dynamics and observed data characteristics.

    References
    ----------
    If the function is based on published research or algorithms, cite these sources here.

    See Also
    --------
    Mention related functions or classes here. For example, functions for sampling unconditional predictive distributions
    or functions for parameter estimation within the same model framework.

    """

    from copy import deepcopy

    # Input type and value checks
    if not isinstance(idata, _az.InferenceData):
        raise TypeError("idata must be an arviz inference data.")

    if not (isinstance(X_test, _np.ndarray) and X_test.ndim == 2):
        raise TypeError("X_test must be a 2D numpy array.")

    if not isinstance(rng_seed, (int, _np.int_)):
        raise TypeError("rng_seed must be an integer.")

    if group not in ['posterior', 'prior']:
        raise ValueError("group must be either 'posterior' or 'prior'.")

    if bootstrap is not None and not isinstance(bootstrap, (int, _np.int_)):
        raise TypeError("bootstrap must be an integer or None.")

    if required not in ['predictive', 'predictive idiosyncratic', 'predictive estimate']:
        raise ValueError("required must be one of 'predictive', 'predictive idiosyncratic', or 'predictive estimate'.")

    # Compatibility checks
    if 'constant_data' in idata.groups():
        if X_test.shape[0] != idata.constant_data['r'].values[0]:
            raise ValueError(f"X_test.shape[0] must match the declared number of covariates")            
    else:
        raise ValueError("idata does not contain 'constant_data', required for shape compatibility checks.")


    rng = _np.random.default_rng(rng_seed)

    if group == "posterior":
        xr_dataset = deepcopy( idata.posterior.stack(sample = ('chain','draw')) )
    elif group == "prior":
        xr_dataset = deepcopy( idata.prior )
    else:
        raise ValueError("group must be one of 'prior' or 'posterior'")

    if not bootstrap is None:
        xr_dataset = xr_dataset.isel(sample=rng.integers(xr_dataset.sizes['sample'], size=bootstrap))

    Ntest = X_test.shape[1]

    estimate_Y1 = \
    _xrein.linalg.matmul(
        renaming_convention( xr_dataset['regr_coeffs1'] ),
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['target_1_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    ).rename({'target_1_dim_idx':'target_dim_idx'})

    estimate_Y2 = \
    _xrein.linalg.matmul(
        renaming_convention( xr_dataset['regr_coeffs2'] ),
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['target_2_dim_idx','covariate_idx'],['covariate_idx','sample_idx']]
    ).rename({'target_2_dim_idx':'target_dim_idx'})

    estimate_Y = \
    _xr.concat(
        [
            estimate_Y1,
            estimate_Y2,
        ],
        dim='target_dim_idx'
    )

    if required == 'predictive estimate':
        # Correctly format the predictive estimate
        Y_predictive = estimate_Y
    else:
        if required == 'predictive':
            Σ = \
            _xr.concat(
                [
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_11'] ).rename({'target_1_dim_idx_bis':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_12'] ).rename({'target_2_dim_idx':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_1_dim_idx':'target_dim_idx'}),
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_21'] ).rename({'target_1_dim_idx':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_22'] ).rename({'target_2_dim_idx_bis':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_2_dim_idx':'target_dim_idx'})
                ],
                dim='target_dim_idx'
            ).rename('Sigma')
        else:
            Σ = \
            _xr.concat(
                [
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_11'] - _np.square( xr_dataset['psi1']*_xr.DataArray(0.99*_np.eye(xr_dataset['Sigma_11'].sizes['Sigma_11_dim_0']), dims=['Sigma_11_dim_0','Sigma_11_dim_1']) ) ).rename({'target_1_dim_idx_bis':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_12'] ).rename({'target_2_dim_idx':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_1_dim_idx':'target_dim_idx'}),
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_21'] ).rename({'target_1_dim_idx':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_22'] - _np.square( xr_dataset['psi2']*_xr.DataArray(0.99*_np.eye(xr_dataset['Sigma_22'].sizes['Sigma_22_dim_0']), dims=['Sigma_22_dim_0','Sigma_22_dim_1']) ) ).rename({'target_2_dim_idx_bis':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_2_dim_idx':'target_dim_idx'})
                ],
                dim='target_dim_idx'
            ).rename('Sigma')

        # Random sample to get the predictive/predictive idiosyncratic
        Y_predictive = estimate_Y + \
        _xrein.linalg.matmul(
            _xrein.linalg.cholesky(
                Σ,
                dims=['target_dim_idx','target_dim_idx2']
            ),
            _xr.DataArray(
                rng.normal(size=tuple( [estimate_Y.sizes[val] for val in estimate_Y.sizes] )),
                dims=[val for val in estimate_Y.sizes]
            ),
            dims=[['target_dim_idx','target_dim_idx2'],['target_dim_idx','sample_idx']]
        )

    if (bootstrap is None) and (group == "posterior"):
        res = Y_predictive.unstack('sample').transpose('chain','draw','target_dim_idx','sample_idx')
    else:
        res = Y_predictive

    return(
        _xr.Dataset(
            {
                'Y1': res.isel(target_dim_idx=_np.arange(idata.constant_data['L1'].values[0])).rename({'target_dim_idx':'target_1_dim_idx'}),
                'Y2': res.isel(target_dim_idx=_np.arange(idata.constant_data['L1'].values[0],
                                                        idata.constant_data['L1'].values[0] + idata.constant_data['L2'].values[0])).rename({'target_dim_idx':'target_2_dim_idx'})
            }
        )
    )


def get_relationship_between_targets(idata, group = "posterior", pointwise = False, bootstrap = None, known_target = 1, rng_seed = 999):
    """
    The function computes the regression coefficients (β) that best describe the relationship between a known 
    target variable and an unknown target variable, based on the covariance matrices provided in the InferenceData 
    object. The method can operate on either the posterior or prior distributions and allows for pointwise 
    regression coefficient calculation.

    Parameters
    ----------
    idata : arviz.InferenceData
        An InferenceData object containing the posterior or prior groups with covariance matrices.
    
    group : str, optional
        The group within the InferenceData to use ('posterior' or 'prior'). Default is 'posterior'.
    
    pointwise : bool, optional
        If True, calculates pointwise regression coefficients. If False, calculates the complete
        regression coefficients matrix. Default is False.
    
    bootstrap : int, optional
        The number of bootstrap samples to use for resampling. If None, no bootstrap resampling is performed.
        Default is None.
    
    known_target : int, optional
        The index of the known target variable (1 or 2). Default is 1.
    
    rng_seed : int, optional
        The seed for the random number generator used in bootstrap resampling. Default is 999. Note that rng_seed 
        is used only if bootstrap is not None.

    Raises
    ------
    TypeError
        If any of the inputs are not of the expected type.
    
    ValueError
        If 'group' is not one of 'posterior' or 'prior', or if 'known_target' is not 1 or 2.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The calculated regression coefficients (β) as an xarray object. If bootstrap is None and group is 
        'posterior', returns an xarray.Dataset; otherwise, returns an xarray.DataArray.

    Notes
    -----
    The regression coefficients are calculated using the covariance matrices between the known and unknown 
    target variables. If 'pointwise' is False, the inverse of the covariance matrix of the known target is used 
    in the computation, otherwise its diagonal is used (This describes the regression from a known `point` of the 
    "known" to the unknown target).

    References
    ----------
    For theoretical background and algorithmic details, refer to relevant literature on Bayesian inference 
    and regression analysis in factor models.

    See Also
    --------
    arviz.InferenceData : Class for storing sampled data for Bayesian inference.
    xarray.DataArray : Data structure for multi-dimensional arrays.
    xarray.Dataset : Data structure for multi-dimensional datasets.

    """
    from copy import deepcopy
    
    if not isinstance(rng_seed, (int, _np.int_)):
        raise TypeError("rng_seed must be an integer.")

    rng = _np.random.default_rng(rng_seed)

    # Input type and value checks
    if not isinstance(idata, _az.InferenceData):
        raise TypeError("idata must be an arviz inference data.")

    if not isinstance(rng_seed, (int, _np.int_)):
        raise TypeError("rng_seed must be an integer.")

    if not isinstance(known_target, (int, _np.int_)) or known_target < 1 or known_target > 2:
        raise TypeError("known_target must be either 1 or 2.")
    if known_target == 1:
        unknown_target = 2
    else:
        unknown_target = 1

    if group not in ['posterior', 'prior']:
        raise ValueError("group must be either 'posterior' or 'prior'.")

    if bootstrap is not None and not isinstance(bootstrap, (int, _np.int_)):
        raise TypeError("bootstrap must be an integer or None.")

    if group == "posterior":
        xr_dataset = deepcopy( idata.posterior.stack(sample = ('chain','draw')) )
    elif group == "prior":
        xr_dataset = deepcopy( idata.prior )
    else:
        raise ValueError("group must be one of 'prior' or 'posterior'")

    if not bootstrap is None:
        xr_dataset = xr_dataset.isel(sample=rng.integers(xr_dataset.sizes['sample'], size=bootstrap))


    Σout_in = renaming_convention( xr_dataset[f'Sigma_{unknown_target}{known_target}'] )
    Σin = renaming_convention( xr_dataset[f'Sigma_{known_target}{known_target}'] )
    Σin = Σin.rename({old:new for old, new in zip(sorted(Σin.dims), ['sample','target_in_idx','target_in_idx2'])})
    Σout_in = Σout_in.rename({f'target_{unknown_target}_dim_idx':'target_out_idx', f'target_{known_target}_dim_idx':'target_in_idx'})

    if pointwise:
        β = (Σout_in/_xrein.linalg.diagonal(
            Σin,
            dims=['target_in_idx','target_in_idx2']
        )).rename({'target_out_idx':f'target_{unknown_target}_dim_idx', 'target_in_idx':f'target_{known_target}_dim_idx'})
    else:
        inverseΣin = \
        _xrein.linalg.inv(
            Σin,
            dims=['target_in_idx','target_in_idx2']
        )

        β = _xrein.linalg.matmul(
            Σout_in,
            inverseΣin,
            dims=[['target_out_idx','target_in_idx'],['target_in_idx2','target_in_idx']]
        ).rename({'target_out_idx':f'target_{unknown_target}_dim_idx', 'target_in_idx':f'target_{known_target}_dim_idx'})

    if (bootstrap is None) and (group == "posterior"):
        return( β.unstack('sample').transpose('chain','draw',f'target_{unknown_target}_dim_idx',f'target_{known_target}_dim_idx') )

    return(β)




def _Varimax_RSP(Lambda):
    '''
    Lambda should be a numpy array in which the first dimension is the MCMC draw, the second is the basis function and the third is the latent factor
    '''
    import scipy.optimize as spopt

    def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
        """Return rotated components."""
        nrow, ncol = components.shape
        rotation_matrix = _np.eye(ncol)
        var = 0
    
        for _ in range(max_iter):
            comp_rot = _np.dot(components, rotation_matrix)
            if method == "varimax":
                tmp = comp_rot * _np.transpose((comp_rot**2).sum(axis=0) / nrow)
            elif method == "quartimax":
                tmp = 0
            u, s, v = _np.linalg.svd(_np.dot(components.T, comp_rot**3 - tmp))
            rotation_matrix = _np.dot(u, v)
            var_new = _np.sum(s)
            if var != 0 and var_new < var * (1 + tol):
                break
            var = var_new
    
        return _np.dot(components, rotation_matrix), rotation_matrix
    
    T = Lambda.shape[0]
    p = Lambda.shape[1]
    q = Lambda.shape[2]
    Λ = _np.zeros( Lambda.shape )
    Phi = _np.zeros( (T,q,q) )
    for t in range(T):
        Λ[t, :, :], Phi[t,:,:] = _ortho_rotation(Lambda[t, :, :])
        if t % 500 == 0:
            print('Rotated sample ' + str(t))        

    S = _np.ones((T, q))
    ν = _np.repeat(
        _np.arange(q)[_np.newaxis, :],
        T,
        axis=0
    )
    
    all_s = _np.array( list(product([-1,1], repeat=q)) )

    objective_fun = +_np.inf

    iteration_number = 0
    while True:
        print('Starting iteration number ' + str(iteration_number))
        iteration_number += 1
        objective_fun_new = 0
        Λstar = _np.zeros((p,q))
        for t in range(T):
            Λaux = (Λ[t,:,:] @ _np.diag(S[t,:]))
            for i in range(q):
                Λstar[:,i] += Λaux[:,ν[t,i]]/T
        for t in range(T):
            ν_new = _np.zeros( all_s.shape )
            cost = _np.zeros( all_s.shape[0] )
            for s_idx in range(all_s.shape[0]):
                s = all_s[s_idx,:]
                Λaux = \
                _np.matmul(
                    Λ[t,:,:],
                    _np.diag(s)
                )
                C = _np.zeros((q,q))
                for j in range(q):
                    C[:,j] = _np.square( Λstar[:,:] - Λaux[:,[j]] ).sum(axis=0)
                row_idx, col_idx = spopt.linear_sum_assignment(C)
                cost[s_idx] = C[row_idx, col_idx].sum()
                ν_new[s_idx, row_idx] = col_idx
            s_idx_best = _np.argmin(cost)
            ν[t,:] = ν_new[s_idx_best, :]
            S[t,:] = all_s[s_idx_best, :]
            objective_fun_new += min(cost)
        if _np.isclose(objective_fun_new, objective_fun, rtol=1e-4):
            break
        if iteration_number >= 50:
            break
        print('\t Previous objective fun =\t' + "{:.{}f}".format(objective_fun, 3) + '\n\t New objective fun =\t\t' + "{:.{}f}".format(objective_fun_new, 3))
        objective_fun = objective_fun_new

    for t in range(T):
        Λ[t,:,:] = Λ[t,:,:] @ _np.diag(S[t,:])
        Λ[t,:,:] = Λ[t,:,ν[t,:]].T
    
    return Phi, Λ, ν, S


def Varimax_RSP(idata):
    """
    Performs Varimax rotation on the posterior samples of the latent factors.
    It combines the latent factors for two targets into a single array, applies the Varimax rotation,
    then looks for the best signed permutation to align all samples about the same mode.
    Returns a dataset containing the rotated latent factors, rotation matrix, sign adjustments, and permutation matrix.
    The function also returns all the paramters that change after the Rotation + Signed Permutation.

    Parameters
    ----------
    idata : az.InferenceData
        The inference Data Object returned from the function `sample_from_posterior`.

    Returns
    -------
    xr.Dataset
        An xarray dataset containing the rotation matrix ('R'), sign adjustments ('S'), 
        permutation matrix ('P'), rotated latent factors for target 1 and 2 ('Λ1', 'Λ2'), 
        the matrix product of basis functions and rotated latent factors for both targets 
        ('B1Λ1', 'B2Λ2'), and the rotated regression coefficients ('β').

    Notes
    -----
    The Varimax rotation is applied to the latent factors to achieve a simpler structure with 
    the goal of making the factors more interpretable. This involves rotating the factors such 
    that the variance of the squared loadings is maximized. The rotation is applied to the 
    combined latent factors for both targets and is followed by adjustments for signs and 
    permutations to align with the original model structure.

    References
    ----------
    Kaiser, H.F. (1958). The varimax criterion for analytic rotation in factor analysis. 
    Psychometrika, 23(3), 187-200.

    This code implements the algorithm described in:
        Papastamoulis, Panagiotis, & Ntzoufras, Ioannis. (2022). On the identifiability of Bayesian factor analytic models. 
        Statistics and Computing, 32(2), 23. https://doi.org/10.1007/s11222-022-10084-4

    See Also
    --------
    _Varimax_RSP : The lower-level function that directly applies the Varimax rotation and additional 
                   transformations to the numpy array of latent factors.

    """
    
    # Validate idata
    if not isinstance(idata, _az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData object.")

    aux_xr = _xr.concat(
        [
            renaming_convention( idata.posterior['Lambda1'] ).rename({'basis_fun_branch_1_idx':'basis_fun_branch_idx'}),
            renaming_convention( idata.posterior['Lambda2'] ).rename({'basis_fun_branch_2_idx':'basis_fun_branch_idx'}),
        ],
        'basis_fun_branch_idx'
    ).stack(chain_draw_idx = ('chain','draw')).transpose('chain_draw_idx','basis_fun_branch_idx','latent_factor_idx')

    Phi, Λ, ν, S = _Varimax_RSP(aux_xr.values)
    
    Phi_xr = \
    _xr.DataArray(
        Phi,
        dims=['chain_draw_idx','latent_factor_idx','latent_factor_idx_bis'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx','latent_factor_idx_bis')

    p1 = idata.constant_data['p1'].values[0]
    p2 = idata.constant_data['p2'].values[0]

    Λ1_xr = \
    _xr.DataArray(
        Λ[:,0:p1,:],
        dims=['chain_draw_idx','basis_fun_branch_1_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','basis_fun_branch_1_idx','latent_factor_idx')

    Λ2_xr = \
    _xr.DataArray(
        Λ[:,p1:(p1+p2),:],
        dims=['chain_draw_idx','basis_fun_branch_2_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','basis_fun_branch_2_idx','latent_factor_idx')

    P = _np.zeros((ν.shape[0],ν.shape[1],ν.shape[1]))
    for t in range(ν.shape[0]):
        P[t,_np.arange(ν.shape[1]),ν[t]] = 1

    # warning! bis comes first because we are hiding a transposition here
    P_xr = \
    _xr.DataArray(
        P,
        dims=['chain_draw_idx','latent_factor_idx_bis','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx','latent_factor_idx_bis')

    S_xr = \
    _xr.DataArray(
        S,
        dims=['chain_draw_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx')

    B1Λ1_xr = \
    _xrein.linalg.matmul(
        renaming_convention( idata.constant_data['B1'] ),
        Λ1_xr,
        dims=[['target_1_dim_idx','basis_fun_branch_1_idx'],['basis_fun_branch_1_idx','latent_factor_idx']]
    )    

    B2Λ2_xr = \
    _xrein.linalg.matmul(
        renaming_convention( idata.constant_data['B2'] ),
        Λ2_xr,
        dims=[['target_2_dim_idx','basis_fun_branch_2_idx'],['basis_fun_branch_2_idx','latent_factor_idx']]
    )    

    Varimaxed_β_xr = \
    _xrein.linalg.matmul(
        P_xr,
        _xrein.linalg.matmul(
            Phi_xr,
            renaming_convention( idata.posterior['beta'] ),
            dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','covariate_idx']]
        ).rename({'latent_factor_idx_bis':'latent_factor_idx'})*S_xr,
        dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','covariate_idx']]
    ).rename({'latent_factor_idx_bis':'latent_factor_idx'})    

    return(
        _xr.Dataset(
            {
                'R': Phi_xr.rename('Rotation Matrix'),
                'S': S_xr.rename('Signed'),
                'P': P_xr.rename('Permutation Matrix'),
                'Λ1': Λ1_xr.rename('Λ target 1'),
                'Λ2': Λ2_xr.rename('Λ target 2'),
                'B1Λ1': B1Λ1_xr,
                'B2Λ2': B2Λ2_xr,
                'β': Varimaxed_β_xr,
            }
        )
    )



def sample_projection_on_varimaxed_space(rng_seed, idata, Varimax_res_xr, X_test = None, Y1_test = None, Y2_test = None, ):
    """
    Projects the training/test samples in the space of Varimax rotated posterior latent factors, considering
    either the training dataset or a provided test dataset. This projection is key for predictions and interpretations
    in the rotated space. When test data (`X_test`) is provided, it computes the projection for this new data. Optionally,
    it can incorporate observations (`Y1_test`, `Y2_test`) to refine the projection.

    Parameters
    ----------
    rng_seed : int
        Seed for the random number generator, ensuring reproducibility.
    idata : az.InferenceData
        The inference data object resulting from sample_from_posterior.
    Varimax_res_xr : xr.Dataset
        The dataset containing the results of the Varimax rotation applied to the latent factors, including
        the rotation matrix, sign adjustments, and permutation matrix.
    X_test : np.ndarray, optional
        A 2D array of the test set covariates. If provided, the function computes the projection on the test data.
        Otherwise, it defaults to None and the function uses the training set for projection.
    Y1_test : np.ndarray, optional
        A 2D array of observations for the first target in the test set. It is used to adjust the projection
        based on observed values. Defaults to None.
    Y2_test : np.ndarray, optional
        A 2D array of observations for the second target in the test set. It is used alongside `Y1_test` to adjust
        the projection based on observed values. Defaults to None.

    Returns
    -------
    xr.DataArray or xr.Dataset
        If `X_test` is None, returns an xarray DataArray containing the projected latent factors for the training set.
        If `X_test` is provided, returns an xarray Dataset with two items: `η_expected` and `η_predictive`, representing
        the expected and predictive projections of the latent factors for the test set, respectively. The predictive
        projections account for uncertainty in the predictions.

    Notes
    -----
    The η obtained from the posterior block of the Inference Data are used when studying the training dataset, in this case
    this function only applies the Varimax correction to the latent scores, when a test dataset is provided, appropriate projections
    are needed first, depending on which information is provided the result will have differing accuracies.

    See Also
    --------
    Varimax_RSP : Function that performs Varimax rotation on the posterior samples of latent factors.

    References
    ----------
    Kaiser, H.F. (1958). The varimax criterion for analytic rotation in factor analysis. Psychometrika, 23(3), 187-200.
    Papastamoulis, Panagiotis, & Ntzoufras, Ioannis. (2022). On the identifiability of Bayesian factor analytic models. 
    Statistics and Computing, 32(2), 23. https://doi.org/10.1007/s11222-022-10084-4

    """

    # Validate rng_seed
    if not isinstance(rng_seed, (int, _np.int_)):
        raise ValueError("rng_seed must be an integer.")
    
    # Validate idata
    if not isinstance(idata, _az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData object.")
    
    # Validate Varimax_res_xr
    if not isinstance(Varimax_res_xr, _xr.Dataset):
        raise ValueError("Varimax_res_xr must be an xarray.Dataset object.")
    
    # Validate X_test, if provided
    if X_test is not None:
        if not isinstance(X_test, _np.ndarray) or X_test.ndim != 2:
            raise ValueError("X_test must be a 2D numpy array.")
    
    # Validate Y1_test, if provided
    if Y1_test is not None:
        if not isinstance(Y1_test, _np.ndarray) or Y1_test.ndim != 2:
            raise ValueError("Y1_test must be a 2D numpy array.")
    
    # Validate Y2_test, if provided
    if Y2_test is not None:
        if not isinstance(Y2_test, _np.ndarray) or Y2_test.ndim != 2:
            raise ValueError("Y2_test must be a 2D numpy array.")

    if X_test is not None:
        # Check if 'r' is in idata.constant_data
        if 'r' not in idata.constant_data:
            raise ValueError("'r' is missing in idata.constant_data.")
        if X_test.shape[0] != idata.constant_data['r'].values[0]:
            raise ValueError(f"X_test's first dimension must match idata.constant_data['r'] = {idata.constant_data['r'].values[0]}.")

    if Y1_test is not None:
        # Ensure X_test is provided when Y1_test is provided
        if X_test is None:
            raise ValueError("X_test must be provided if Y1_test is specified.")
        # Check if 'L1' is in idata.constant_data
        if 'L1' not in idata.constant_data:
            raise ValueError("'L1' is missing in idata.constant_data.")
        if Y1_test.shape[0] != idata.constant_data['L1'].values[0]:
            raise ValueError(f"Y1_test's first dimension must match idata.constant_data['L1'] = {idata.constant_data['L1'].values[0]}.")
        if X_test.shape[1] != Y1_test.shape[1]:
            raise ValueError("X_test and Y1_test must have the same number of columns (samples).")

    if Y2_test is not None:
        # Ensure X_test is provided when Y2_test is provided
        if X_test is None:
            raise ValueError("X_test must be provided if Y2_test is specified.")
        # Check if 'L2' is in idata.constant_data
        if 'L2' not in idata.constant_data:
            raise ValueError("'L2' is missing in idata.constant_data.")
        if Y2_test.shape[0] != idata.constant_data['L2'].values[0]:
            raise ValueError(f"Y2_test's first dimension must match idata.constant_data['L2'] = {idata.constant_data['L2'].values[0]}.")
        if X_test.shape[1] != Y2_test.shape[1]:
            raise ValueError("X_test and Y2_test must have the same number of columns (samples).")        

    from copy import deepcopy

    rng = _np.random.default_rng(rng_seed)

    if X_test is None:
        # the order ['latent_factor_idx_bis','latent_factor_idx'] implies a transposition of the rotation matrix
        res_xr = \
        _xrein.linalg.matmul(
            Varimax_res_xr['P'],
            _xrein.linalg.matmul(
                Varimax_res_xr['R'],
                renaming_convention( idata.posterior['eta'] ),
                dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','sample_idx']]
            ).rename({'latent_factor_idx_bis':'latent_factor_idx'})*Varimax_res_xr['S'],
            dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','sample_idx']]
        ).rename({'latent_factor_idx_bis':'latent_factor_idx'})
        return( res_xr )

    xr_dataset = deepcopy( idata.posterior )

    η_estimate = \
    _xrein.linalg.matmul(
        renaming_convention( xr_dataset['beta'] ),
        _xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['latent_factor_idx','covariate_idx'],['covariate_idx','sample_idx']]
    )

    Σ = _xr.DataArray(_np.eye(η_estimate.sizes['latent_factor_idx']), dims=['latent_factor_idx','latent_factor_idx2'])
    aux_xr = sample_unconditional_predictive(idata, X_test, 13, required='predictive estimate')

    if (Y1_test is None) and (Y2_test is None):
        n_provided_Ys = 0
    elif (not Y1_test is None) and (not Y2_test is None):
        n_provided_Ys = 2
        Y1_test_xr = _xr.DataArray(
            Y1_test,
            dims=['target_1_dim_idx','sample_idx']
        )
        Y2_test_xr = _xr.DataArray(
            Y2_test,
            dims=['target_2_dim_idx','sample_idx']
        )
        ΔY_xr = \
        _xr.concat(
            [
                renaming_convention( Y1_test_xr ).rename({'target_1_dim_idx':'target_dim_idx'}),
                renaming_convention( Y2_test_xr ).rename({'target_2_dim_idx':'target_dim_idx'})
            ],
            dim='target_dim_idx'
        ) - _xr.concat(
            [
                aux_xr['Y1'].rename({'target_1_dim_idx':'target_dim_idx'}),
                aux_xr['Y2'].rename({'target_2_dim_idx':'target_dim_idx'})
            ],
            dim='target_dim_idx'
        )

        invΣy = \
        _xrein.linalg.inv(
            _xr.concat(
                [
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_11'] ).rename({'target_1_dim_idx_bis':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_12'] ).rename({'target_2_dim_idx':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_1_dim_idx':'target_dim_idx'}),
                    _xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_21'] ).rename({'target_1_dim_idx':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_22'] ).rename({'target_2_dim_idx_bis':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_2_dim_idx':'target_dim_idx'})
                ],
                dim='target_dim_idx'
            ).rename('Sigma'),
            dims=['target_dim_idx','target_dim_idx2']
        )

        Σyη = \
        _xr.concat(
            [
                Varimax_res_xr['B1Λ1'].rename({'target_1_dim_idx':'target_dim_idx'}),
                Varimax_res_xr['B2Λ2'].rename({'target_2_dim_idx':'target_dim_idx'})
            ],
            dim='target_dim_idx'
        )

    else:
        n_provided_Ys = 1
        if not Y1_test is None:
            # Y1 is provided Y2 not
            Y1_test_xr = _xr.DataArray(
                Y1_test,
                dims=['target_1_dim_idx','sample_idx']
            )
            g = 1
            ΔY_xr = Y1_test_xr.rename({'target_1_dim_idx':'target_dim_idx'}) - aux_xr['Y1'].rename({'target_1_dim_idx':'target_dim_idx'})
        else:
            # Y2 is provided Y1 not
            Y2_test_xr = _xr.DataArray(
                Y2_test,
                dims=['target_2_dim_idx','sample_idx']
            )
            g = 2
            ΔY_xr = Y2_test_xr.rename({'target_2_dim_idx':'target_dim_idx'}) - aux_xr['Y2'].rename({'target_2_dim_idx':'target_dim_idx'})
        invΣy = _xrein.linalg.inv(
            renaming_convention( xr_dataset[f'Sigma_{g}{g}'] ).rename({f'target_{g}_dim_idx':'target_dim_idx',f'target_{g}_dim_idx_bis':'target_dim_idx2'}),
            dims=['target_dim_idx','target_dim_idx2']
        )
        Σyη = Varimax_res_xr[f'B{g}Λ{g}'].rename({f'target_{g}_dim_idx':'target_dim_idx'})


    if not ((Y1_test is None) and (Y2_test is None)):
        η_estimate += \
        _xrein.linalg.matmul(
            Σyη,
            _xrein.linalg.matmul(
                invΣy,
                ΔY_xr,
                dims=[['target_dim_idx','target_dim_idx2'],['target_dim_idx','sample_idx']]
            ),
            dims=[['latent_factor_idx','target_dim_idx'],['target_dim_idx','sample_idx']]
        )

        Σ = \
        _xrein.linalg.matmul(
            Σyη,
            _xrein.linalg.matmul(
                invΣy,
                Σyη,
                dims=[['target_dim_idx','target_dim_idx2'],['target_dim_idx','latent_factor_idx']]
            ),
            dims=[['latent_factor_idx','target_dim_idx'],['target_dim_idx','latent_factor_idx']]
        ) - Σ
        Σ = -Σ

    CholeskyΣ = _xrein.linalg.cholesky(
        Σ,
        dims=['latent_factor_idx','latent_factor_idx2']
    )

    η_predictive = deepcopy(η_estimate)
    η_predictive += \
    _xrein.linalg.matmul(
        CholeskyΣ,
        _xr.DataArray(
            rng.normal(size = η_estimate.shape),
            dims=η_estimate.dims
        ),
        dims=[['latent_factor_idx','latent_factor_idx2'],['latent_factor_idx','sample_idx']]
    )

    """
    return(
        xr.Dataset(
            {
                'η_expected': expected_η_xr,
                'η_predictive': η_predictive,
            }
        )
    )
    """

    return(
        η_predictive
    )


