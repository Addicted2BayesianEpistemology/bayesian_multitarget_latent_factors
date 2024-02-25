

import numpy as np
import arviz as az
import xarray as xr
import xarray_einstats as xrein
import skfda.representation.basis as scikit_basis_funs
from .multitarget_latent_factor_model import sample_from_prior


__all__ = ['default_basis_dictionary_builder', 'dataset_generator', 'one_functional_target_dictionary_builder',
			'compactly_supported_radial_basis_fun', 'generate_2D_grid', 'compute_euclidean_distances', 'dataset_generator_2D_domain']


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

    import warnings
    warnings.warn("Functoin left for backcompatiblity")
    
    rng = np.random.default_rng(rng_seed)
    data_dic = default_basis_dictionary_builder(
        rng.integers(1000000),
        N = n_train_samples + n_test_samples,
        r = r, L1 = L1, domain_range_1 = domain_range_1, p1 = p1,
        L2 = L2, domain_range_2 = domain_range_2, p2 = p2, k = k,
        X_scale_multiplier = X_scale_multiplier, nu = nu, v = v,
        alpha_1 = alpha_1, alpha_2 = alpha_2, alpha_psi = alpha_psi,
        alpha_sigma = alpha_sigma,
        psi_noise_scale_multiplier_1 = psi_noise_scale_multiplier_1, psi_noise_scale_multiplier_2 = psi_noise_scale_multiplier_2,
        theta_noise_scale_multiplier_1 = theta_noise_scale_multiplier_1, theta_noise_scale_multiplier_2 = theta_noise_scale_multiplier_2
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
    xr.Dataset(
        {var: idata_simul[var].sel(sample=0) for var in idata_simul.data_vars}
    ).drop(['log_lik_y','y1_predictive','y2_predictive','y1_test_predictive','y2_test_predictive'])

    aux_xr['X_test'] = xr.DataArray(X_test, dims=['X_test_dim_0','X_test_dim_1'])
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
    import warnings
    warnings.warn("Functoin left for backcompatiblity")

    # Validate input parameters
    if N <= 0 or r <= 0 or L1 <= 0 or L2 <= 0 or  p2 <= 0 or k <= 0:
        raise ValueError("N, r, L1, L2, p2, and k must be positive integers.")
    if (not isinstance(N , (int, np.int_))) or (not isinstance(r, (int, np.int_))) or (not isinstance(L1, (int, np.int_))) or (not isinstance(L2, (int, np.int_))) or (not isinstance(p2, (int, np.int_))) or (not isinstance(k, (int, np.int_))):
        raise ValueError("N, r, L1, L2, p2, and k must be positive integers.")
    if domain_range_2[0] >= domain_range_2[1]:
        raise ValueError("Domain range must have a lower bound less than the upper bound.")
    if not isinstance(domain_range_2, list) or len(domain_range_2) != 2:
        raise ValueError("Domain ranges must be lists of two elements.")
    if nu <= 0 or v <= 0 or alpha_1 <= 0 or alpha_2 <= 0 or alpha_psi <= 0 or beta_psi <= 0 or alpha_sigma <= 0 or beta_sigma <= 0:
        raise ValueError("nu, v, alpha_1, alpha_2, alpha_psi, beta_psi, alpha_sigma, and beta_sigma must be positive.")
    if not isinstance(rng_seed, (int, np.int_)):
        raise ValueError("rng_seed must be an integer")    

    # Initialize the random number generator
    rng = np.random.default_rng(rng_seed)
    
    # The vector of sampling locations
    t1 = np.arange(L1)
    t2 = np.linspace(domain_range_2[0], domain_range_2[1], L2)

    B1 = np.eye(L1,L1)
    Bspline2 = \
    scikit_basis_funs.BSplineBasis(
        domain_range=domain_range_2,
        n_basis=p2-1,
        order=3
    )
    B2 = \
    np.concatenate(
        [
            np.ones((L2,1)),
            Bspline2(t2)[:,:,0].T
        ],
        axis=1
    )
    
    data_dic = {
        'N': N, 'L1': L1, 'L2': L2, 'p1': L1, 'p2': p2, 'k': k, 'r': r,
        'y1': np.zeros((L1, N)), 'y2': np.zeros((L2, N)),
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
    import warnings
    warnings.warn("Functoin left for backcompatiblity")

    # Validate input parameters
    if N <= 0 or r <= 0 or L1 <= 0 or L2 <= 0 or p1 <= 0 or p2 <= 0 or k <= 0:
        raise ValueError("N, r, L1, L2, p1, p2, and k must be positive integers.")
    if (not isinstance(N , (int, np.int_))) or (not isinstance(r, (int, np.int_))) or (not isinstance(L1, (int, np.int_))) or (not isinstance(L2, (int, np.int_))) or (not isinstance(p1, (int, np.int_))) or (not isinstance(p2, (int, np.int_))) or (not isinstance(k, (int, np.int_))):
        raise ValueError("N, r, L1, L2, p1, p2, and k must be positive integers.")
    if domain_range_1[0] >= domain_range_1[1] or domain_range_2[0] >= domain_range_2[1]:
        raise ValueError("Domain range must have a lower bound less than the upper bound.")
    if not isinstance(domain_range_1, list) or not isinstance(domain_range_2, list) or len(domain_range_1) != 2 or len(domain_range_2) != 2:
        raise ValueError("Domain ranges must be lists of two elements.")
    if nu <= 0 or v <= 0 or alpha_1 <= 0 or alpha_2 <= 0 or alpha_psi <= 0 or beta_psi <= 0 or alpha_sigma <= 0 or beta_sigma <= 0:
        raise ValueError("nu, v, alpha_1, alpha_2, alpha_psi, beta_psi, alpha_sigma, and beta_sigma must be positive.")
    if not isinstance(rng_seed, (int, np.int_)):
        raise ValueError("rng_seed must be an integer")    

    # Initialize the random number generator
    rng = np.random.default_rng(rng_seed)
    
    # The vector of sampling locations
    t1 = np.linspace(domain_range_1[0], domain_range_1[1], L1)
    t2 = np.linspace(domain_range_2[0], domain_range_2[1], L2)

    
    # The basis matrices are made up of B-Splines, plus a constant column (like in the original model)
    Bspline1 = \
    scikit_basis_funs.BSplineBasis(
        domain_range=domain_range_1,
        n_basis=p1-1,
        order=3
    )
    B1 = \
    np.concatenate(
        [
            np.ones((L1,1)),
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
    np.concatenate(
        [
            np.ones((L2,1)),
            Bspline2(t2)[:,:,0].T
        ],
        axis=1
    )
    
    data_dic = {
        'N': N, 'L1': L1, 'L2': L2, 'p1': p1, 'p2': p2, 'k': k, 'r': r,
        'y1': np.zeros((L1, N)), 'y2': np.zeros((L2, N)),
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



def radial_basis_function(r):
    """
    Thin-plate spline radial basis function.
    
    Parameters:
    r (ndarray): Euclidean distances between points.
    
    Returns:
    ndarray: Evaluated thin-plate spline function.
    """
    # Use np.where to avoid log(0) which is undefined.
    return np.where(r == 0, 0, r**2 * np.log(r))


def build_tps_basis_matrix(evaluation_points, center_points):
    """
    Build the basis matrix for thin-plate splines given two sets of points.
    
    Parameters:
    evaluation_points (ndarray): An Mx2 array of 2D points where functions are evaluated.
    center_points (ndarray): An Nx2 array of 2D points where the splines are centered.
    
    Returns:
    ndarray: The Mx(N+3) basis matrix for thin-plate splines.

    Example:
    evaluation_points = np.array([[0.5, 0.5], [1.5, 1.5], [1.5, 0.5], [0.5, 1.5]])
    center_points = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    basis_matrix = build_tps_basis_matrix(evaluation_points, center_points)
    print(basis_matrix)
    """
    # Compute the pairwise distances between evaluation points and center points
    r = np.sqrt(np.sum((evaluation_points[:, np.newaxis, :] - center_points[np.newaxis, :, :]) ** 2, axis=-1))
    # Apply the radial basis function
    basis_matrix = radial_basis_function(r)
    
    # Append ones and linear terms to the evaluation points to account for the affine part
    M = evaluation_points.shape[0]
    P = np.hstack((np.ones((M, 1)), evaluation_points))
    basis_matrix = np.hstack((basis_matrix, P))
    
    if center_points.shape[0] > 0:
        # Only append zeros to P if there are center points
        P_extended = np.hstack((P, np.zeros((P.shape[0], center_points.shape[0]))))
        basis_matrix = np.vstack((basis_matrix, np.zeros((3, basis_matrix.shape[1]))))
    
    return basis_matrix


def compactly_supported_radial_basis_fun(r, ε):
    """
    Bump function

    Parameters:
    r (ndarray): Euclidean distances between points.

    Retunrs:
    ndarray: Evaluated bump functions
    """
    return np.where(r*ε >= 1, 0, np.exp(-1/(1 - np.square(r*ε))))


def compute_euclidean_distances(evaluation_points, center_points):
    # Convert lists to numpy arrays if they aren't already
    evaluation_points = np.array(evaluation_points)
    center_points = np.array(center_points)
    
    # Calculate the differences between each evaluation point and center point
    # This involves expanding both arrays to 3 dimensions to leverage broadcasting for pairwise differences
    diff = evaluation_points[:, np.newaxis, :] - center_points[np.newaxis, :, :]
    
    # Compute the Euclidean distances using the norm function along the last axis (coordinates axis)
    distances = np.linalg.norm(diff, axis=-1)
    
    return distances

def generate_2D_grid(x_start, x_end, x_num, y_start, y_end, y_num):
    import warnings
    warnings.warn("Functoin left for backcompatiblity")
    # Generate linearly spaced points for x and y axes
    x_eval_points = np.linspace(x_start, x_end, x_num)
    y_eval_points = np.linspace(y_start, y_end, y_num)
    
    # Create a meshgrid for x and y points
    X, Y = np.meshgrid(x_eval_points, y_eval_points)
    
    # Reshape and stack the X and Y coordinates into a two-column array
    eval_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    return eval_points


def dataset_generator_2D_domain(rng_seed,
                                n_train_samples = 50,
                                n_test_samples = 20,
                                r = 5,
                                L1 = 30,
                                domain_range_1 = [0,1],
                                p1 = 10,
                                L2_x_axis = 10,
                                L2_y_axis = 10,
                                domain_range_2_x_axis = [0,3],
                                domain_range_2_y_axis = [0,3],
                                p2_x_axis = 3,
                                p2_y_axis = 4,
                                ε = 2,
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
    TO-DO CORRECT THE DOCSTRING
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

    ε : float, optional
        Inverse radius of the bump radial function
        
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
    import warnings
    warnings.warn("Functoin left for backcompatiblity")
    
    rng = np.random.default_rng(rng_seed)

    locs = generate_2D_grid(domain_range_2_x_axis[0], domain_range_2_x_axis[1], L2_x_axis,
                            domain_range_2_y_axis[0], domain_range_2_y_axis[1], L2_y_axis)

    basis_locations = generate_2D_grid(domain_range_2_x_axis[0], domain_range_2_x_axis[1], p2_x_axis,
                                       domain_range_2_y_axis[0], domain_range_2_y_axis[1], p2_y_axis)


    B2 = \
    np.concatenate(
        [
            np.ones((locs.shape[0], 1)),
            locs,
            locs.prod(axis=1)[:,np.newaxis],
            compactly_supported_radial_basis_fun(
                compute_euclidean_distances(
                    locs,
                    basis_locations
                ),
                ε
            )
        ],
        axis=1
    )

    data_dic = default_basis_dictionary_builder(
        rng.integers(1000000),
        N = n_train_samples + n_test_samples,
        p2 = B2.shape[1],
        L2 = B2.shape[0],
    )

    data_dic['B2'] = B2
    data_dic['t2'] = locs

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
    xr.Dataset(
        {var: idata_simul[var].sel(sample=0) for var in idata_simul.data_vars}
    ).drop(['log_lik_y','y1_predictive','y2_predictive','y1_test_predictive','y2_test_predictive'])

    aux_xr['X_test'] = xr.DataArray(X_test, dims=['X_test_dim_0','X_test_dim_1'])
    aux_xr['y1'] = y1_test
    aux_xr['y2'] = y2_test


    return data_dic, aux_xr









# from here on everything is new
__all__ += ['make_prior_dict','plot_basis_column']

from skfda.representation.basis import BSpline, ConstantBasis, Monomial, Fourier, TensorBasis
import matplotlib.pyplot as plt



def _add_optional_terms_to_basis(B, add_optional_terms, dimensionality, domain_range, grid_spacing, num_points):
    """
    Adds optional 'constant', 'linear', or 'bilinear' terms to the basis matrix B to enhance model flexibility.

    This function allows for the inclusion of additional terms to the basis functions, which can be useful for
    capturing trends that are not well-represented by the original basis alone. This can include constant terms for
    intercepts, linear terms for trends, and bilinear terms for interactions in two-dimensional data.

    Parameters
    ----------
    B : numpy.ndarray
        The evaluated basis matrix to which the optional terms will be added.
    add_optional_terms : str
        Option to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Should be one of
        'constant', 'linear', 'bilinear', or 'no' to indicate no additional terms should be added.
    dimensionality : int
        The dimensionality of the data (1 or 2) indicating whether the basis functions are for one-dimensional
        or two-dimensional data.
    domain_range : tuple or list of tuples
        The domain range(s) for the basis function(s). For 1D data, this is a single tuple specifying the start
        and end of the domain. For 2D data, a list of two tuples or a single tuple used for both dimensions.
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points. For 1D data, this should be a single float. For 2D data, this can be a tuple
        of two floats specifying the spacing for each dimension.
        If a single value is provided, it is replicated for both dimensions when dimensionality is 2.
        If None, `num_points` will be used to determine the grid points instead.
    num_points : int or tuple of ints, optional
        The number of points for the evaluation grid. For 1D data, this should be a single integer. For 2D data, this
        can be a tuple of two integers specifying the number of points for each dimension.
        If a single value is provided, it is replicated for both dimensions when dimensionality is 2.
        If `grid_spacing` is provided, `num_points` is ignored.

    Returns
    -------
    numpy.ndarray
        The modified basis matrix with optional terms added, if specified.

    Raises
    ------
    ValueError
        If `add_optional_terms` is not one of 'constant', 'linear', 'bilinear', or 'no'.
        If `dimensionality` is not 1 or 2.
        If `domain_range` does not match the expected format based on `dimensionality`.

    Notes
    -----
    The function directly modifies and returns the input basis matrix `B` by appending columns for the optional
    terms specified. The domain range is used to calculate linear terms evenly spaced over the specified domain.

    See Also
    --------
    _setup_bspline_basis : Function to set up a B-spline basis.
    _setup_fourier_basis : Function to set up a Fourier basis.

    """
    # Validation checks
    if add_optional_terms not in ['constant', 'linear', 'bilinear', 'no']:
        raise ValueError("`add_optional_terms` must be one of 'constant', 'linear', 'bilinear', or 'no'.")
    
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")
    
    if dimensionality == 1 and not isinstance(domain_range, tuple):
        raise ValueError("For 1D data, `domain_range` must be a tuple specifying the start and end of the domain.")
    elif dimensionality == 2:
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Use the same range for both dimensions
        elif not isinstance(domain_range, list) or len(domain_range) != 2 or not all(isinstance(r, tuple) for r in domain_range):
            raise ValueError("For 2D data, `domain_range` must be a list of two tuples specifying the domain ranges.")

    if add_optional_terms == 'bilinear' and dimensionality == 1:
        raise ValueError("bilinear is implemented only for the 2D data case")

    if add_optional_terms in ['constant','linear', 'bilinear']:
        # Add a column of ones to represent the constant term
        if dimensionality == 1:
            constant_column = np.ones((B.shape[0], 1))/np.sqrt(domain_range[1] - domain_range[0])
        else:
            constant_column = np.ones((B.shape[0], 1))/np.sqrt((domain_range[0][1] - domain_range[0][0])*(domain_range[1][1] - domain_range[1][0]))
        B = np.hstack((B, constant_column))
    if add_optional_terms in ['linear', 'bilinear']:
        # For 1D, add a column that linearly increases across the domain range
        if dimensionality == 1:
            linear_column = np.linspace(domain_range[0], domain_range[1], B.shape[0]).reshape(-1, 1)
            linear_column = linear_column - 0.5*(domain_range[0] + domain_range[1])
            linear_column = np.sqrt(12)*linear_column/np.power(domain_range[1] - domain_range[0], 1.5)
            B = np.hstack((B, linear_column))
        # For 2D, add two columns, one for each dimension
        else:
            linear_column_x, linear_column_y = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
            linear_column_x = linear_column_x.flatten() - 0.5*(domain_range[0][0] + domain_range[0][1])
            linear_column_x = np.sqrt(12)*linear_column_x/np.power(domain_range[0][1] - domain_range[0][0], 1.5)
            linear_column_y = linear_column_y.flatten() - 0.5*(domain_range[1][0] + domain_range[1][1])
            linear_column_y = np.sqrt(12)*linear_column_y/np.power(domain_range[1][1] - domain_range[1][0], 1.5)
            B = np.hstack((B, linear_column_x.reshape(-1, 1), linear_column_y.reshape(-1, 1)))
            if add_optional_terms == 'bilinear':
                bilinear_column = (linear_column_x * linear_column_y).reshape(-1, 1)
                B = np.hstack((B, bilinear_column))
    return B


def _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality):
    """
    Determines the evaluation grid points based on domain range, grid spacing, number of points, and dimensionality.

    This function computes a set of grid points used for evaluating basis functions over a specified domain. It supports
    both one-dimensional and two-dimensional data. For one-dimensional data, it generates a linearly spaced array. For
    two-dimensional data, it generates a meshgrid of points covering the domain.

    Parameters
    ----------
    domain_range : tuple or list of tuples
        The domain range(s) for evaluating the basis functions. For 1D, this should be a tuple specifying the start
        and end of the domain. For 2D, it should be a list of two tuples, each specifying the start and end of the
        domain along each dimension. A single tuple is automatically duplicated for 2D data.
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points. For 1D data, this should be a single float. For 2D data, this can be a tuple
        of two floats specifying the spacing for each dimension. A single float is automatically duplicated for 2D data.
        If None, `num_points` will be used to determine the grid points instead.
    num_points : int or tuple of ints, optional
        The number of points for the evaluation grid. For 1D data, this should be a single integer. For 2D data, this
        can be a tuple of two integers specifying the number of points for each dimension. A single integer is automatically
        duplicated for 2D data.
        If `grid_spacing` is provided, `num_points` is ignored.
    dimensionality : int
        The dimensionality of the data (1 or 2).

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        The evaluation grid points. For 1D data, this is a numpy array. For 2D data, this is a list of two numpy arrays
        representing the meshgrid.

    Raises
    ------
    ValueError
        If `dimensionality` is not 1 or 2.
        If `domain_range` does not match the expected format based on `dimensionality`.
        If `grid_spacing` or `num_points` are not provided in a compatible format for the specified dimensionality.

    Notes
    -----
    The function intelligently decides between using `grid_spacing` and `num_points` to generate the grid points, prioritizing
    `grid_spacing` if it is provided. It ensures that the grid covers the entire domain range specified.

    See Also
    --------
    _add_optional_terms_to_basis : Function to add optional terms to a basis matrix.
    """
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")

    if dimensionality == 1:
        if not isinstance(domain_range, tuple) or len(domain_range) != 2:
            raise ValueError("For 1D data, `domain_range` must be a tuple of two elements specifying the start and end of the domain.")
    else:  # dimensionality == 2
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]
        elif not isinstance(domain_range, list) or len(domain_range) != 2 or not all(isinstance(r, tuple) and len(r) == 2 for r in domain_range):
            raise ValueError("For 2D data, `domain_range` must be a list of two tuples, each specifying the domain range for one dimension.")

    if dimensionality == 2:
        if isinstance(grid_spacing, (float, int)):
            grid_spacing = (grid_spacing, grid_spacing)
        if isinstance(num_points, int):
            num_points = (num_points, num_points)

    if grid_spacing is not None and not (isinstance(grid_spacing, (float, tuple)) or (isinstance(grid_spacing, list) and all(isinstance(g, float) for g in grid_spacing))):
        raise ValueError("`grid_spacing` must be a float, or a tuple/list of floats for 2D data.")

    # Ensure either `grid_spacing` or `num_points` is provided
    if grid_spacing is None and num_points is None:
        raise ValueError("Either `grid_spacing` or `num_points` must be provided.")
    
    if not isinstance(num_points, (int, tuple, list)) or (isinstance(num_points, tuple) and len(num_points) != 2 and dimensionality == 2):
        raise ValueError("`num_points` must be an integer for 1D data or a tuple of two integers for 2D data.")
    
    if dimensionality == 1:
        if grid_spacing is not None:
            num_points = int((domain_range[1] - domain_range[0]) / grid_spacing) + 1
        grid_points = np.linspace(domain_range[0], domain_range[1], num_points)
    else:  # dimensionality == 2
        if grid_spacing is not None:
            if isinstance(grid_spacing, float):
                num_points_x = int((domain_range[0][1] - domain_range[0][0]) / grid_spacing) + 1
                num_points_y = int((domain_range[1][1] - domain_range[1][0]) / grid_spacing) + 1
            else:  # grid_spacing is a tuple
                num_points_x = int((domain_range[0][1] - domain_range[0][0]) / grid_spacing[0]) + 1
                num_points_y = int((domain_range[1][1] - domain_range[1][0]) / grid_spacing[1]) + 1
        else:  # Use num_points directly
            if isinstance(num_points, int):
                num_points_x, num_points_y = num_points
            else: # num_points is a tuple
                num_points_x = num_points[0]
                num_points_y = num_points[1]
        grid_x = np.linspace(domain_range[0][0], domain_range[0][1], num_points_x)
        grid_y = np.linspace(domain_range[1][0], domain_range[1][1], num_points_y)
        grid_points = np.meshgrid(grid_x, grid_y, indexing='ij')
    return grid_points


def _setup_bspline_basis(dimensionality, p, domain_range, n_basis):
    """
    Sets up a B-spline basis for use in nonparametric functional data analysis, supporting both one-dimensional
    and two-dimensional data.

    This function initializes and returns a B-spline basis object, which is essential for constructing the
    functional data basis. The B-spline basis can be used to represent complex data structures through a
    series of basis functions, allowing for smooth approximations of the data.

    Parameters
    ----------
    dimensionality : int
        The dimensionality of the data (1 or 2), indicating whether the basis should be set up for one-dimensional
        or two-dimensional data.
    p : int or tuple of ints
        The order of the B-spline. For 1D data, this is a single integer. For 2D data, this can be either a single
        integer (applied to both dimensions) or a tuple of two integers, specifying the order for each dimension.
    domain_range : tuple or list of tuples
        The domain range over which the basis functions are defined. For 1D data, this is a tuple specifying
        the start and end of the domain. For 2D data, this can be either a single tuple (applied to both dimensions)
        or a list of two tuples, each specifying the domain range for one dimension.
    n_basis : int or tuple of ints
        The number of basis functions to use. For 1D data, this is a single integer. For 2D data, this can be either
        a single integer (applied to both dimensions) or a tuple of two integers, specifying the number of basis
        functions for each dimension.

    Returns
    -------
    BSpline or TensorBasis
        A B-spline basis object for 1D data or a tensor basis composed of two B-spline basis objects for 2D data.

    Raises
    ------
    ValueError
        If `dimensionality` is not 1 or 2.
        If `p` or `n_basis` do not match the expected format based on `dimensionality`.
        If `domain_range` does not match the expected format based on `dimensionality`.

    Notes
    -----
    The B-spline basis is a flexible tool for functional data analysis, allowing for smooth and adaptable
    representations of the underlying data structure. This setup function is a crucial step in preparing
    the basis for modeling and analysis.
    The function is designed to automatically adapt single input values for `p`, `n_basis`, and `domain_range`
    for 2D configurations, enhancing usability and flexibility.

    See Also
    --------
    BSpline : Class for creating B-spline basis functions.
    TensorBasis : Class for creating tensor products of basis functions for multidimensional data.
    """
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")
    
    if dimensionality == 1:
        if not isinstance(p, int) or not isinstance(n_basis, int):
            raise ValueError("For 1D data, `p` and `n_basis` must be integers.")
        if not isinstance(domain_range, tuple) or len(domain_range) != 2:
            raise ValueError("For 1D data, `domain_range` must be a tuple of two elements specifying the start and end of the domain.")
    else:  # dimensionality == 2
        if isinstance(p, int):
            p = (p, p)
        if isinstance(n_basis, int):
            n_basis = (n_basis, n_basis)
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]
        if not (isinstance(p, tuple) and len(p) == 2 and all(isinstance(order, int) for order in p)):
            raise ValueError("For 2D data, `p` must be a tuple of two integers specifying the order for each dimension.")
        if not (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis)):
            raise ValueError("For 2D data, `n_basis` must be a tuple of two integers specifying the number of basis functions for each dimension.")
        if not (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range)):
            raise ValueError("For 2D data, `domain_range` must be a list of two tuples, each specifying the domain range for one dimension.")
    
    if dimensionality == 1:
        basis = BSpline(domain_range=domain_range, n_basis=n_basis, order=p)
    else:  # dimensionality == 2
        basis_x = BSpline(domain_range=domain_range[0], n_basis=n_basis[0], order=p[0])
        basis_y = BSpline(domain_range=domain_range[1], n_basis=n_basis[1], order=p[1])
        basis = TensorBasis([basis_x, basis_y])
    return basis

def _make_prior_dict_basis_builder_bspline(dimensionality=1, add_optional_terms='no', p=4, domain_range=(0, 1), n_basis=10, grid_spacing=None, num_points=10):
    """
    Constructs and returns a basis matrix for B-spline basis functions, incorporating optional terms and allowing
    for the specification of evaluation grid by either grid spacing or number of points.

    This function facilitates the creation of a basis matrix B, pivotal for modeling in nonparametric functional
    data analysis. It supports the addition of 'constant' and 'linear' terms to the basis functions, enhancing
    the model's flexibility and capability to capture underlying data structures across one-dimensional and
    two-dimensional spaces.

    Parameters
    ----------
    dimensionality : int, optional
        The dimensionality of the data (1 or 2). Defaults to 1.
        If dimensionality is 2 and any parameter is provided as a single value instead of a tuple or list,
        the single value will be automatically expanded to apply uniformly across both dimensions.
    add_optional_terms : str, optional
        Indicates whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    p : int or tuple of ints
        The order of the B-spline. For 1D data, this is a single integer. For 2D data, this should be a tuple
        of two integers, specifying the order for each dimension.
    domain_range : tuple, optional
        Specifies the domain range over which the basis functions are defined. For 1D data, it is a tuple of
        the form (start, end). For 2D data, a list of such tuples. Defaults to (0, 1).
    n_basis : int, optional
        The number of basis functions to use. Defaults to 10.
    grid_spacing : float or tuple of floats, optional
        Defines the spacing between grid points for evaluating the basis functions. If specified, `num_points`
        is ignored. For 2D data, it can be a tuple specifying the spacing for each dimension.
    num_points : int or tuple of ints, optional
        Determines the number of points for evaluating the basis functions. Used only if `grid_spacing` is not
        provided. For 2D data, it can be a tuple specifying the number of points for each dimension. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        The evaluated and optionally modified basis matrix B, suitable for further analysis or model fitting.

    Raises
    ------
    ValueError
        Raised if any of the input parameters do not conform to their expected formats or allowable values.

    Notes
    -----
    The function seamlessly integrates various steps necessary for preparing the basis matrix for B-spline basis
    functions, including the setup of the basis according to specified parameters, evaluation of the basis on a
    determined grid, and the optional addition of terms to the basis for enhanced modeling capabilities.

    See Also
    --------
    _setup_bspline_basis : Function to set up a B-spline basis.
    _determine_evaluation_grid : Function to determine the grid points for basis function evaluation.
    _add_optional_terms_to_basis : Function to add optional terms to the basis matrix.
    """
    from skfda.misc.metrics import LpNorm

    # Validation checks (Examples, additional checks may be required based on function implementation)
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")

    # Validation and expansion for `n_basis`
    if isinstance(n_basis, int):
        n_basis = (n_basis, n_basis) if dimensionality == 2 else n_basis
    elif dimensionality == 2 and (not isinstance(n_basis, tuple) or len(n_basis) != 2):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation and expansion for `p`
    if isinstance(p, int):
        p = (p, p) if dimensionality == 2 else p
    elif dimensionality == 2 and (not isinstance(p, tuple) or len(p) != 2):
        raise ValueError("`p` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation and expansion for `domain_range`
    if dimensionality == 1:
        if not isinstance(domain_range, tuple):
            raise ValueError("`domain_range` must be a tuple for 1D data.")
    else:  # dimensionality == 2
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Expand to list of tuples for 2D
        elif not (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) for dr in domain_range)):
            raise ValueError("`domain_range` must be a tuple or a list of two tuples for 2D data.")

    # Validation and expansion for `grid_spacing`
    if grid_spacing is not None:
        if isinstance(grid_spacing, (float, int)):
            grid_spacing = (grid_spacing, grid_spacing) if dimensionality == 2 else grid_spacing
        elif dimensionality == 2 and (not isinstance(grid_spacing, tuple) or len(grid_spacing) != 2):
            raise ValueError("`grid_spacing` must be a float or a tuple of two floats for 2D data.")

    # Validation and expansion for `num_points`
    if isinstance(num_points, int):
        num_points = (num_points, num_points) if dimensionality == 2 else num_points
    elif dimensionality == 2 and (not isinstance(num_points, tuple) or len(num_points) != 2):
        raise ValueError("`num_points` must be an integer for 1D data or a tuple of two integers for 2D data.")

    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be 'no', 'constant', 'linear', or 'bilinear'.")
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")
    if not ((isinstance(domain_range, tuple) and len(domain_range) == 2 and dimensionality == 1) or (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range) and dimensionality == 2)):
        raise ValueError("`domain_range` must be a tuple for 1D data or a list of two tuples for 2D data.")

    basis = _setup_bspline_basis(dimensionality, p, domain_range, n_basis)

    # Determine evaluation grid
    grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)

    if dimensionality == 2:
        aux = np.array(grid_points)
        grid_points = np.reshape( aux , (2, np.prod( aux.shape[1:] ))).T

    # Evaluate basis
    B = basis(grid_points)

    B = B[:,:,0].T
    B = B/np.repeat(LpNorm(2)(basis)[np.newaxis,:], B.shape[0], axis=0)

    # Add optional terms to B
    B = _add_optional_terms_to_basis(B, add_optional_terms, dimensionality, domain_range, grid_spacing, num_points)

    return B


def _setup_fourier_basis(dimensionality, n_basis, domain_range):
    """
    Initializes and returns a Fourier basis for modeling, applicable to both one-dimensional and two-dimensional data.

    This function configures a Fourier basis, which is critical for representing periodic components within the data
    through harmonic functions. It supports constructing the basis for either univariate or multivariate data, allowing
    for flexible modeling approaches in functional data analysis.

    Parameters
    ----------
    dimensionality : int
        The dimensionality of the data (either 1 or 2), indicating whether the basis should be configured for
        one-dimensional or two-dimensional data.
    n_basis : int or tuple of ints
        The number of basis functions to include in the Fourier basis. If a single integer is provided for 2D data,
        it will be interpreted as having the same number of basis functions for each dimension.
    domain_range : tuple or list of tuples
        The domain range(s) over which the Fourier basis functions are defined. For 1D data, this is a tuple
        specifying the start and end of the domain. For 2D data, a single tuple can be provided, which will be
        applied to both dimensions, or a list of two tuples, each specifying the domain range for one dimension.

    Returns
    -------
    Fourier or TensorBasis
        A Fourier basis object for 1D data or a tensor basis composed of two Fourier basis objects for 2D data.

    Raises
    ------
    ValueError
        If `dimensionality` is not 1 or 2.
        If `n_basis` does not match the expected format based on `dimensionality`.
        If `domain_range` does not match the expected format based on `dimensionality`.

    Notes
    -----
    The Fourier basis provides a method for capturing periodic patterns within the data, making it an essential
    component for models that require the representation of cyclical behavior. The setup process ensures that
    the basis is correctly configured for the specific needs of the analysis, including the dimensionality and
    domain of the data.

    See Also
    --------
    Fourier : Class for creating Fourier basis functions.
    TensorBasis : Class for creating tensor products of basis functions for multidimensional data.
    """
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")

    # Handle n_basis for 2D data when provided as an integer
    if dimensionality == 2 and isinstance(n_basis, int):
        n_basis = (n_basis, n_basis)  # Duplicate the value for both dimensions

    # Adjust domain_range for 2D data when provided as a single tuple
    if dimensionality == 2 and isinstance(domain_range, tuple):
        domain_range = [domain_range, domain_range]  # Apply the tuple to both dimensions

    if dimensionality == 1 and not isinstance(n_basis, int):
        raise ValueError("For 1D data, `n_basis` must be an integer.")
    elif dimensionality == 2 and (not isinstance(n_basis, tuple) or len(n_basis) != 2 or not all(isinstance(nb, int) for nb in n_basis)):
        raise ValueError("For 2D data, `n_basis` must be a tuple of two integers specifying the number of basis functions for each dimension.")

    if dimensionality == 1 and (not isinstance(domain_range, tuple) or len(domain_range) != 2):
        raise ValueError("For 1D data, `domain_range` must be a tuple specifying the start and end of the domain.")
    elif dimensionality == 2 and (not isinstance(domain_range, list) or len(domain_range) != 2 or not all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range)):
        raise ValueError("For 2D data, `domain_range` must be a list of two tuples, each specifying the domain range for one dimension.")
    
    if dimensionality == 1:
        basis = Fourier(n_basis=n_basis, domain_range=domain_range)
    else:  # dimensionality == 2
        basis_x = Fourier(n_basis=n_basis[0], domain_range=domain_range[0])
        basis_y = Fourier(n_basis=n_basis[1], domain_range=domain_range[1])
        basis = TensorBasis([basis_x, basis_y])
    return basis



def _make_prior_dict_basis_builder_fourier(dimensionality=1, add_optional_terms='no', n_basis=10, domain_range=(0, 1), grid_spacing=None, num_points=10):
    """
    Constructs and returns a basis matrix for Fourier basis functions, including optional terms and allowing
    for grid spacing or number of points specification for the evaluation grid.

    This function enables the creation of a Fourier basis matrix B, essential for modeling periodic behaviors
    in data through harmonic functions. It offers the flexibility to add 'constant' and 'linear' terms to the
    basis functions, enhancing the model's adaptability for one-dimensional and two-dimensional data analysis.

    Parameters
    ----------
    dimensionality : int, optional
        The dimensionality of the data (either 1 or 2), indicating whether the basis is for one-dimensional
        or two-dimensional data. Defaults to 1.
    add_optional_terms : str, optional
        Specifies whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    n_basis : int or tuple of ints, optional
        The number of basis functions for the Fourier basis. For 1D data, this is a single integer. For 2D data,
        if a single integer is provided, it is used for both dimensions; a tuple of two integers indicates the
        number of basis functions for each dimension. Defaults to 10.
    domain_range : tuple or list of tuples, optional
        The domain range(s) over which the basis functions are defined. For 1D data, a tuple of (start, end).
        For 2D data, if a single tuple is provided, it is applied to both dimensions; a list of two tuples, each
        specifying the domain range for one dimension. Defaults to (0, 1).
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points for evaluating the basis functions. Ignored if `num_points` is specified.
        For 2D data, if a single float is provided, it is applied to both dimensions; a tuple specifies the spacing
        for each dimension.
    num_points : int or tuple of ints, optional
        The number of points for the basis function evaluation. If `grid_spacing` is not provided, this parameter
        is used. For 2D data, if a single integer is provided, it is used for both dimensions; a tuple specifies the
        number of points for each dimension. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        The evaluated basis matrix B, after adding optional terms if specified.

    Raises
    ------
    ValueError
        If any of the input parameters are not in their expected formats or if their values are not permissible.

    Notes
    -----
    By incorporating optional terms and providing flexibility in the evaluation grid setup, this function extends
    the capabilities of Fourier basis for modeling. It is integral for capturing and representing periodic patterns
    within the data, tailored to the specific requirements of the analysis.

    See Also
    --------
    _setup_fourier_basis : Function to initialize a Fourier basis.
    _determine_evaluation_grid : Function to determine evaluation grid points.
    _add_optional_terms_to_basis : Function to enhance the basis matrix with additional terms.
    """
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")

    # Validation and adaptation for domain_range
    if dimensionality == 2:
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Use the same range for both dimensions
        elif not (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) for dr in domain_range)):
            raise ValueError("`domain_range` must be a tuple for 1D data or a list of two tuples for 2D data.")

    # Validation and adaptation for n_basis
    if dimensionality == 2:
        if isinstance(n_basis, int):
            n_basis = (n_basis, n_basis)  # Use the same number of basis functions for both dimensions
        elif not (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis)):
            raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation and adaptation for grid_spacing
    if dimensionality == 2 and not grid_spacing is None:
        if isinstance(grid_spacing, float):
            grid_spacing = (grid_spacing, grid_spacing)  # Apply the same spacing for both dimensions
        elif not (isinstance(grid_spacing, tuple) and len(grid_spacing) == 2 and all(isinstance(gs, (float, type(None))) for gs in grid_spacing)):
            raise ValueError("`grid_spacing` must be a float or None for 1D data, or a tuple of two floats/None for 2D data.")

    # Validation and adaptation for num_points
    if dimensionality == 2:
        if isinstance(num_points, int):
            num_points = (num_points, num_points)  # Use the same number of points for both dimensions
        elif not (isinstance(num_points, tuple) and len(num_points) == 2 and all(isinstance(np, int) for np in num_points)):
            raise ValueError("`num_points` must be an integer for 1D data or a tuple of two integers for 2D data.")

    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be 'no', 'constant', 'linear', or 'bilinear'.")
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")
    if not ((isinstance(domain_range, tuple) and len(domain_range) == 2 and dimensionality == 1) or (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range) and dimensionality == 2)):
        raise ValueError("`domain_range` must be a tuple for 1D data or a list of two tuples for 2D data.")

    # Validate inputs and setup basis according to dimensionality and optional terms
    basis = _setup_fourier_basis(dimensionality, n_basis, domain_range)

    # Determine evaluation grid
    grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)

    if dimensionality == 2:
        aux = np.array(grid_points)
        grid_points = np.reshape( aux , (2, np.prod( aux.shape[1:] ))).T

    # Evaluate basis
    B = basis(grid_points)

    B = B[1:,:,0].T  # 1: because we remove the only constant term (which is common between the 1D and 2D case)

    # Add optional terms to B
    B = _add_optional_terms_to_basis(B, add_optional_terms, dimensionality, domain_range, grid_spacing, num_points)

    return B


def _make_prior_dict_basis_builder_radial_bump(dimensionality=2, add_optional_terms='no', R=1.0, domain_range=(0, 1), n_basis=10, grid_spacing=None, num_points=10):
    """
    Constructs and returns a basis matrix for radial basis function, specifically using a bump function defined
    by exp(-1/(1-(r/R)^2)), with the inclusion of optional terms and allowing for the specification of evaluation
    grid by either grid spacing or number of points.

    This function is designed for nonparametric functional data analysis, supporting both one-dimensional and
    two-dimensional data. It allows for a flexible specification of the evaluation grid and includes the option
    to enhance the model's adaptability with 'constant', 'linear', or 'bilinear' terms.

    Parameters
    ----------
    dimensionality : int, optional
        Specifies the dimensionality of the data (either 1 or 2), defaulting to 2. For 2D data, single inputs
        for parameters like `domain_range`, `n_basis`, etc., are expanded to apply uniformly across dimensions.
    add_optional_terms : str, optional
        Indicates whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    R : float, or tuple of floats
        The radius of the bump radial function, must be positive. If a tuple of floats is provided, the normalization
        along the two axis is different.
    domain_range : tuple or list of tuples, optional
        Specifies the domain range over which the basis functions are defined. For 1D data, it is a tuple of
        the form (start, end). For 2D data, a list of two tuples or a single tuple to be used for both dimensions.
        Defaults to (0, 1).
    n_basis : int or tuple of ints, optional
        The number of basis functions to use, defaulting to 10. For 2D data, can be a tuple specifying the number
        of basis functions for each dimension or a single integer to apply uniformly.
    grid_spacing : float or tuple of floats, optional
        Defines the spacing between grid points for evaluating the basis functions. If specified, `num_points`
        is ignored. For 2D data, it can be a tuple specifying the spacing for each dimension or a single float.
    num_points : int or tuple of ints, optional
        Determines the number of points for evaluating the basis functions, defaulting to 10. For 2D data, can be
        a tuple specifying the number of points for each dimension or a single integer to apply uniformly.

    Returns
    -------
    numpy.ndarray
        The evaluated and optionally modified basis matrix B, suitable for further analysis or model fitting.

    Raises
    ------
    ValueError
        Raised if any of the input parameters do not conform to their expected formats or allowable values.

    Notes
    -----
    The function seamlessly integrates various steps necessary for preparing the basis matrix for radial basis bump
    functions, including the setup of the basis according to specified parameters, evaluation of the basis on a
    determined grid, and the optional addition of terms to the basis for enhanced modeling capabilities.

    See Also
    --------
    _setup_bspline_basis : Function to set up a B-spline basis.
    _determine_evaluation_grid : Function to determine the grid points for basis function evaluation.
    _add_optional_terms_to_basis : Function to add optional terms to the basis matrix.
    """
    # Validation checks (Examples, additional checks may be required based on function implementation)
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")
    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be 'no', 'constant', 'linear', or 'bilinear'.")

    # Validation check for R
    if dimensionality == 1:
        if not (isinstance(R, (int, float)) and R > 0):
            raise ValueError("`R` must be a positive number for 1D data.")
    elif dimensionality == 2:
        if isinstance(R, (int, float)):
            if R <= 0:
                raise ValueError("`R` must be a positive number for 2D data.")
            R = (R, R)  # Make R uniform across both dimensions
        elif isinstance(R, tuple):
            if len(R) != 2 or not all(isinstance(r, (int, float)) and r > 0 for r in R):
                raise ValueError("For 2D data, `R` must be a tuple of two positive numbers.")
        else:
            raise ValueError("For 2D data, `R` must be a positive number or a tuple of two positive numbers.")

    
    # Handling single inputs for 2D cases
    if dimensionality == 2:
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Duplicate tuple for 2D
        if isinstance(n_basis, int):
            n_basis = (n_basis, n_basis)  # Duplicate value for 2D
        if isinstance(grid_spacing, (int, float)):
            grid_spacing = (grid_spacing, grid_spacing)  # Duplicate value for 2D
        if isinstance(num_points, int):
            num_points = (num_points, num_points)  # Duplicate value for 2D

    # Validate domain_range format
    if dimensionality == 1 and not isinstance(domain_range, tuple):
        raise ValueError("For 1D data, `domain_range` must be a tuple specifying the start and end of the domain.")
    elif dimensionality == 2 and (not isinstance(domain_range, list) or len(domain_range) != 2 or not all(isinstance(r, tuple) for r in domain_range)):
        raise ValueError("For 2D data, `domain_range` must be a list of two tuples specifying the domain ranges.")
    
    # Validate n_basis format
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validate grid_spacing format
    if grid_spacing is not None and not (isinstance(grid_spacing, (int, float)) or (isinstance(grid_spacing, tuple) and len(grid_spacing) == 2 and all(isinstance(gs, (int, float)) for gs in grid_spacing))):
        raise ValueError("`grid_spacing` must be a float or an int, or a tuple of two floats/ints for 2D data.")

    # Validate num_points format
    if not ((isinstance(num_points, int) and dimensionality == 1) or (isinstance(num_points, tuple) and len(num_points) == 2 and all(isinstance(np, int) for np in num_points) and dimensionality == 2)):
        raise ValueError("`num_points` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Determine nodes grid
    nodes_points = _determine_evaluation_grid(domain_range, None, n_basis, dimensionality)

    # Determine evaluation grid
    grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)


    def _compactly_supported_radial_basis_fun(r):
        return np.where(r >= 1.0, 0, np.exp(-1/(1 - np.square(r))))


    def _compute_euclidean_distances_2D(evaluation_points, center_points):
        # Calculate the differences between each evaluation point and center point
        # This involves expanding both arrays to 3 dimensions to leverage broadcasting for pairwise differences
        evaluation_points_internal = np.array(evaluation_points)
        evaluation_points_internal = np.reshape( evaluation_points_internal , (2, np.prod(evaluation_points[0].shape))).T
        center_points_internal = np.array(center_points)
        center_points_internal = np.reshape( center_points_internal , (2, np.prod(center_points[0].shape))).T
        diff = evaluation_points_internal[:, np.newaxis, :] - center_points_internal[np.newaxis, :, :]
        diff[:,:,0] = diff[:,:,0]/R[0]
        diff[:,:,1] = diff[:,:,1]/R[1]
        # Compute the Euclidean distances using the norm function along the last axis (coordinates axis)
        distances = np.linalg.norm(diff, axis=-1)
        return distances

    with np.errstate(all='ignore'):
        if dimensionality == 1:
            delta = grid_points[1] - grid_points[0]
            B = _compactly_supported_radial_basis_fun(
                np.abs(
                    grid_points[:,np.newaxis] - nodes_points[np.newaxis, :]
                )/R
            )
            B = B/np.repeat( np.sqrt( np.square(B).sum(axis=0)*delta )[np.newaxis,:] , B.shape[0], axis=0)
        else:
            delta_x = np.diff( grid_points[0][[0,1],0] )[0]
            delta_y = np.diff( grid_points[1][0,[0,1]] )[0]
            B = _compactly_supported_radial_basis_fun(
                _compute_euclidean_distances_2D(
                    grid_points, nodes_points
                )
            )
            B = B/np.repeat( np.sqrt( np.square(B).sum(axis=0)*(delta_x*delta_y) )[np.newaxis,:] , B.shape[0], axis=0)

    # Add optional terms to B
    B = _add_optional_terms_to_basis(B, add_optional_terms, dimensionality, domain_range, grid_spacing, num_points)

    return B




def plot_basis_column(basis_type='bspline', dimensionality=1, p=4, n_basis = 10, R=1.0, domain_range=(0, 1), grid_spacing=None, num_points=10, column_to_plot=0, add_optional_terms='no'):
    """
    Constructs and plots a specific column of the basis matrix, which is built using either B-spline, Fourier or radial Bump
    basis functions. The plot adapts based on the data's dimensionality, displaying as a 1D line plot or a 2D heatmap.

    This function demonstrates the flexibility of functional data analysis tools in visualizing the components
    of the basis matrix. By selecting a specific column, users can inspect the shape and characteristics of
    individual basis functions, aiding in the understanding and interpretation of the model's behavior.

    Parameters
    ----------
    basis_type : str, optional
        Specifies the type of basis to use: 'bspline' for B-spline basis functions, 'fourier' for Fourier
        basis functions, or 'radial_bump' for radial bump functions. Defaults to 'bspline'.
    dimensionality : int, optional
        The dimensionality of the data (1 or 2), indicating whether the plot should be generated for one-dimensional
        or two-dimensional data. Defaults to 1.
    p : int, optional
        The degree of the B-spline (used when `basis_type` is 'bspline'). Defaults to 4.
    n_basis : int, optional
        The number of basis functions to include in the basis matrix. For `basis_type` 'fourier', this parameter
        is analogous to the number of harmonics. Defaults to 10.
    R : float, optional
        The radius of the bump function, used only when `basis_type` is 'radial_bump'. Must be positive. Defaults to 1.0.
    domain_range : tuple or list of tuples, optional
        The domain range(s) over which the basis functions are evaluated. For 1D data, a tuple of (start, end).
        For 2D data, a list of two tuples, each specifying the domain range for one dimension. Defaults to (0, 1).
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points for evaluating the basis functions. Ignored if `num_points` is specified.
        For 2D data, it can be a tuple specifying the spacing for each dimension.
    num_points : int or tuple of ints, optional
        The number of points for the basis function evaluation. If `grid_spacing` is not provided, this parameter
        is used. For 2D data, it can be a tuple specifying the number of points for each dimension. Defaults to 10.
    column_to_plot : int, optional
        The index of the column of the basis matrix B to plot, allowing inspection of specific basis functions.
        Defaults to 0.
    add_optional_terms : str, optional
        Specifies whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions, enhancing the
        model's adaptability. Defaults to 'no'.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes._axes.Axes
        The figure and axes objects of the plot, enabling further customization or display.

    Raises
    ------
    ValueError
        If `basis_type` is not 'bspline' or 'fourier'.
        If `dimensionality` is not 1 or 2.
        If any other parameter values are outside their expected ranges or formats.

    Notes
    -----
    This function is a utility for visualizing the influence and shape of the basis functions used in modeling,
    providing insights into the model's construction and potential behavior.

    Examples
    --------
    >>> plot_basis_column(basis_type='bspline', dimensionality=2, p=3, n_basis=(10, 10),
                          domain_range=[(0, 1), (0, 1)], column_to_plot=5, add_optional_terms='linear')
    """

    from .plotting_tools import plot_unstructured_heatmap

    # Validation checks
    if basis_type not in ['bspline', 'fourier', 'radial_bump']:
        raise ValueError("`basis_type` must be 'bspline' or 'fourier' or 'radial_bump'.")
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")
    if not isinstance(column_to_plot, int) or column_to_plot < 0:
        raise ValueError("`column_to_plot` must be a non-negative integer.")

    # Build the basis matrix B
    if basis_type == 'bspline':
        B = _make_prior_dict_basis_builder_bspline(dimensionality=dimensionality, add_optional_terms=add_optional_terms, p=p, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'fourier':
        B = _make_prior_dict_basis_builder_fourier(dimensionality=dimensionality, add_optional_terms=add_optional_terms, n_basis=n_basis, domain_range=domain_range, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'radial_bump':
        B = _make_prior_dict_basis_builder_radial_bump(dimensionality=dimensionality, add_optional_terms=add_optional_terms, R=R, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    else:
        raise ValueError("Invalid basis_type. Choose 'bspline', 'fourier', or 'radial_bump'.")

    # Plotting
    if dimensionality == 1:
        ax = plt.gca()
        fig = ax.get_figure()
        fig.set_figwidth(9)
        fig.set_figheight(6)
        x = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
        plt.plot(x, B[:, column_to_plot], label=f'Column {column_to_plot}')
        plt.title(f'Plot of Column {column_to_plot} of Basis Matrix')
        plt.xlabel('X Axis')
        plt.ylabel('Value')
        plt.legend()
        return fig, ax
    elif dimensionality == 2:
        plt.figure(figsize=(8, 6))
        grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
        aux = np.array(grid_points)
        grid_points = np.reshape( aux , (2, np.prod( aux.shape[1:] ))).T
        ax,img,cbar,vmin,vmax = plot_unstructured_heatmap(
            B[:, column_to_plot], grid_points, grid_res=400, colormap = 'coolwarm'
        )

        ax.set_title(f'Heatmap of Column {column_to_plot} of Basis Matrix')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
    else:
        raise ValueError("dimensionality must be either 1 or 2")


def plot_basis_all(basis_type='bspline', dimensionality=1, p=4, n_basis = 10, R=1.0, domain_range=(0, 1), grid_spacing=None, num_points=10, add_optional_terms='no'):
    """
    Constructs and plots the whole basis matrix, which is built using either B-spline, Fourier or radial Bump
    basis functions. The plot adapts based on the data's dimensionality, displaying as a 1D line plot or a 2D heatmap.

    This function demonstrates the flexibility of functional data analysis tools in visualizing the components
    of the basis matrix. By selecting a specific column, users can inspect the shape and characteristics of
    individual basis functions, aiding in the understanding and interpretation of the model's behavior.

    Parameters
    ----------
    basis_type : str, optional
        Specifies the type of basis to use: 'bspline' for B-spline basis functions, 'fourier' for Fourier
        basis functions, or 'radial_bump' for radial bump functions. Defaults to 'bspline'.
    dimensionality : int, optional
        The dimensionality of the data (1 or 2), indicating whether the plot should be generated for one-dimensional
        or two-dimensional data. Defaults to 1.
    p : int, optional
        The degree of the B-spline (used when `basis_type` is 'bspline'). Defaults to 4.
    n_basis : int, optional
        The number of basis functions to include in the basis matrix. For `basis_type` 'fourier', this parameter
        is analogous to the number of harmonics. Defaults to 10.
    R : float, optional
        The radius of the bump function, used only when `basis_type` is 'radial_bump'. Must be positive. Defaults to 1.0.
    domain_range : tuple or list of tuples, optional
        The domain range(s) over which the basis functions are evaluated. For 1D data, a tuple of (start, end).
        For 2D data, a list of two tuples, each specifying the domain range for one dimension. Defaults to (0, 1).
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points for evaluating the basis functions. Ignored if `num_points` is specified.
        For 2D data, it can be a tuple specifying the spacing for each dimension.
    num_points : int or tuple of ints, optional
        The number of points for the basis function evaluation. If `grid_spacing` is not provided, this parameter
        is used. For 2D data, it can be a tuple specifying the number of points for each dimension. Defaults to 10.
    add_optional_terms : str, optional
        Specifies whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions, enhancing the
        model's adaptability. Defaults to 'no'.

    Returns
    -------
    None if Dimensionality == 1
    matplotlib.animation.FuncAnimation object if Dimensionality == 2

    Raises
    ------
    ValueError
        If `basis_type` is not 'bspline' or 'fourier'.
        If `dimensionality` is not 1 or 2.
        If any other parameter values are outside their expected ranges or formats.

    Notes
    -----
    This function is a utility for visualizing the influence and shape of the basis functions used in modeling,
    providing insights into the model's construction and potential behavior.

    Examples
    --------
    >>> from IPython.display import HTML
    >>> HTML(
            bmlf.plot_basis_all(
                basis_type='bspline',
                dimensionality=2,
                p=3,
                n_basis=4,
                R=0.15,
                domain_range=(0, 1),
                grid_spacing=None,
                num_points=100,
                add_optional_terms='linear',
            ).to_jshtml()
        )

    """
    from .plotting_tools import plot_unstructured_heatmap
    from matplotlib.animation import FuncAnimation

    # Validation checks
    if basis_type not in ['bspline', 'fourier', 'radial_bump']:
        raise ValueError("`basis_type` must be 'bspline' or 'fourier' or 'radial_bump'.")
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")

    # Build the basis matrix B
    if basis_type == 'bspline':
        B = _make_prior_dict_basis_builder_bspline(dimensionality=dimensionality, add_optional_terms=add_optional_terms, p=p, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'fourier':
        B = _make_prior_dict_basis_builder_fourier(dimensionality=dimensionality, add_optional_terms=add_optional_terms, n_basis=n_basis, domain_range=domain_range, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'radial_bump':
        B = _make_prior_dict_basis_builder_radial_bump(dimensionality=dimensionality, add_optional_terms=add_optional_terms, R=R, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    else:
        raise ValueError("Invalid basis_type. Choose 'bspline', 'fourier', or 'radial_bump'.")

    # Plotting
    if dimensionality == 1:
        for column_to_plot in range(B.shape[1]):
            ax = plt.gca()
            fig = ax.get_figure()
            fig.set_figwidth(9)
            fig.set_figheight(6)
            x = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
            plt.plot(x, B[:, column_to_plot])
            plt.xlabel('X Axis')
            plt.ylabel('Value')
    elif dimensionality == 2:
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        idx = np.arange(B.shape[1])

        vmin = B.min()
        vmax = B.max()

        grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
        aux = np.array(grid_points)
        grid_points = np.reshape( aux , (2, np.prod( aux.shape[1:] ))).T

        plot_unstructured_heatmap(
            B[:,0],
            grid_points,
            colormap='coolwarm',
            show_colorbar=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax
        )

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        def update(idx):
            plot_unstructured_heatmap(
                B[:,idx],
                grid_points,
                colormap='coolwarm',
                show_colorbar=False,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                title=f'Column {idx} of basis matrix B'
            )
            
        anim = FuncAnimation(fig, update, frames = idx, interval=200)
        return anim
    else:
        raise ValueError("dimensionality must be either 1 or 2")



def _add_optional_terms_to_basis_unstructured(evaluation_coords, B, add_optional_terms, dimensionality, domain_range):
    """
    Adds optional 'constant', 'linear', or 'bilinear' terms to the basis matrix B to enhance model flexibility.

    This function allows for the inclusion of additional terms to the basis functions, which can be useful for
    capturing trends that are not well-represented by the original basis alone. This can include constant terms for
    intercepts, linear terms for trends, and bilinear terms for interactions in two-dimensional data.

    Parameters
    ----------
    evaluation_coords: numpy.array
        A numpy array of shape (n_points, ) in the 1D case or (n_points, 2) in the 2D case
    B : numpy.ndarray
        The evaluated basis matrix to which the optional terms will be added.
    add_optional_terms : str
        Option to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Should be one of
        'constant', 'linear', 'bilinear', or 'no' to indicate no additional terms should be added.
    dimensionality : int
        The dimensionality of the data (1 or 2) indicating whether the basis functions are for one-dimensional
        or two-dimensional data.
    domain_range : tuple or list of tuples
        The domain range(s) for the basis function(s). For 1D data, this is a single tuple specifying the start
        and end of the domain. For 2D data, a list of two tuples or a single tuple used for both dimensions.

    Returns
    -------
    numpy.ndarray
        The modified basis matrix with optional terms added, if specified.

    Raises
    ------
    ValueError
        If `add_optional_terms` is not one of 'constant', 'linear', 'bilinear', or 'no'.
        If `dimensionality` is not 1 or 2.
        If `domain_range` does not match the expected format based on `dimensionality`.
        If `evaluation_coords` does not have the correct shape based on `dimensionality`.

    Notes
    -----
    The function directly modifies and returns the input basis matrix `B` by appending columns for the optional
    terms specified. The domain range is used to calculate linear terms evenly spaced over the specified domain.
    """
    # Validation checks for add_optional_terms and dimensionality are already in place
    
    # Validation check for evaluation_coords
    if not isinstance(evaluation_coords, np.ndarray):
        raise ValueError("`evaluation_coords` must be a numpy array.")
    if dimensionality == 1:
        if not (evaluation_coords.ndim == 1 or (evaluation_coords.ndim == 2 and evaluation_coords.shape[1] == 1)):
            raise ValueError("For 1D data, `evaluation_coords` must have shape (n_points,) or (n_points, 1).")
    elif dimensionality == 2:
        if not (evaluation_coords.ndim == 2 and evaluation_coords.shape[1] == 2):
            raise ValueError("For 2D data, `evaluation_coords` must have shape (n_points, 2).")
    # Validation checks
    if add_optional_terms not in ['constant', 'linear', 'bilinear', 'no']:
        raise ValueError("`add_optional_terms` must be one of 'constant', 'linear', 'bilinear', or 'no'.")
    
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")
    
    if dimensionality == 1 and not isinstance(domain_range, tuple):
        raise ValueError("For 1D data, `domain_range` must be a tuple specifying the start and end of the domain.")
    elif dimensionality == 2:
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Use the same range for both dimensions
        elif not isinstance(domain_range, list) or len(domain_range) != 2 or not all(isinstance(r, tuple) for r in domain_range):
            raise ValueError("For 2D data, `domain_range` must be a list of two tuples specifying the domain ranges.")

    if add_optional_terms == 'bilinear' and dimensionality == 1:
        raise ValueError("bilinear is implemented only for the 2D data case")

    if add_optional_terms in ['constant','linear', 'bilinear']:
        # Add a column of ones to represent the constant term
        if dimensionality == 1:
            constant_column = np.ones((B.shape[0], 1))/np.sqrt(domain_range[1] - domain_range[0])
        else:
            constant_column = np.ones((B.shape[0], 1))/np.sqrt((domain_range[0][1] - domain_range[0][0])*(domain_range[1][1] - domain_range[1][0]))
        B = np.hstack((B, constant_column))
    if add_optional_terms in ['linear', 'bilinear']:
        # For 1D, add a column that linearly increases across the domain range
        if dimensionality == 1:
            linear_column = np.linspace(domain_range[0], domain_range[1], B.shape[0]).reshape(-1, 1)
            linear_column = linear_column - 0.5*(domain_range[0] + domain_range[1])
            linear_column = np.sqrt(12)*linear_column/np.power(domain_range[1] - domain_range[0], 1.5)
            B = np.hstack((B, linear_column))
        # For 2D, add two columns, one for each dimension
        else:
            linear_column_x = evaluation_coords[:,0].copy()
            linear_column_y = evaluation_coords[:,1].copy()
            linear_column_x = linear_column_x.flatten() - 0.5*(domain_range[0][0] + domain_range[0][1])
            linear_column_x = np.sqrt(12)*linear_column_x/np.power(domain_range[0][1] - domain_range[0][0], 1.5)
            linear_column_y = linear_column_y.flatten() - 0.5*(domain_range[1][0] + domain_range[1][1])
            linear_column_y = np.sqrt(12)*linear_column_y/np.power(domain_range[1][1] - domain_range[1][0], 1.5)
            B = np.hstack((B, linear_column_x.reshape(-1, 1), linear_column_y.reshape(-1, 1)))
            if add_optional_terms == 'bilinear':
                bilinear_column = (linear_column_x * linear_column_y).reshape(-1, 1)
                B = np.hstack((B, bilinear_column))
    return B




def _make_prior_dict_basis_builder_bspline_unstructured(evaluation_coords, dimensionality=1, add_optional_terms='no', p=4, domain_range=(0, 1), n_basis=10):
    """
    Constructs and returns a basis matrix for B-spline basis functions, incorporating optional terms and allowing
    unstructured evaluation locations through the parameter evaluation_coords.

    This function facilitates the creation of a basis matrix B, pivotal for modeling in nonparametric functional
    data analysis. It supports the addition of 'constant' and 'linear' terms to the basis functions, enhancing
    the model's flexibility and capability to capture underlying data structures across one-dimensional and
    two-dimensional spaces.

    Parameters
    ----------
    evaluation_coords: numpy.array
        A numpy array of shape (n_points, ) in the 1D case or (n_points, 2) in the 2D case
    dimensionality : int, optional
        The dimensionality of the data (1 or 2). Defaults to 1.
        If dimensionality is 2 and any parameter is provided as a single value instead of a tuple or list,
        the single value will be automatically expanded to apply uniformly across both dimensions.
    add_optional_terms : str, optional
        Indicates whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    p : int or tuple of ints
        The order of the B-spline. For 1D data, this is a single integer. For 2D data, this should be a tuple
        of two integers, specifying the order for each dimension.
    domain_range : tuple, optional
        Specifies the domain range over which the basis functions are defined. For 1D data, it is a tuple of
        the form (start, end). For 2D data, a list of such tuples. Defaults to (0, 1).
    n_basis : int, optional
        The number of basis functions to use. Defaults to 10.

    Returns
    -------
    numpy.ndarray
        The evaluated and optionally modified basis matrix B, suitable for further analysis or model fitting.

    Raises
    ------
    ValueError
        Raised if any of the input parameters do not conform to their expected formats or allowable values.

    Notes
    -----
    The function seamlessly integrates various steps necessary for preparing the basis matrix for B-spline basis
    functions, including the setup of the basis according to specified parameters, evaluation of the basis on a
    determined grid, and the optional addition of terms to the basis for enhanced modeling capabilities.

    See Also
    --------
    _setup_bspline_basis : Function to set up a B-spline basis.
    _determine_evaluation_grid : Function to determine the grid points for basis function evaluation.
    _add_optional_terms_to_basis : Function to add optional terms to the basis matrix.
    """
    from skfda.misc.metrics import LpNorm

    # Validation checks (Examples, additional checks may be required based on function implementation)
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")

    # Validation and expansion for `n_basis`
    if isinstance(n_basis, int):
        n_basis = (n_basis, n_basis) if dimensionality == 2 else n_basis
    elif dimensionality == 2 and (not isinstance(n_basis, tuple) or len(n_basis) != 2):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation and expansion for `p`
    if isinstance(p, int):
        p = (p, p) if dimensionality == 2 else p
    elif dimensionality == 2 and (not isinstance(p, tuple) or len(p) != 2):
        raise ValueError("`p` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation and expansion for `domain_range`
    if dimensionality == 1:
        if not isinstance(domain_range, tuple):
            raise ValueError("`domain_range` must be a tuple for 1D data.")
    else:  # dimensionality == 2
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Expand to list of tuples for 2D
        elif not (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) for dr in domain_range)):
            raise ValueError("`domain_range` must be a tuple or a list of two tuples for 2D data.")

    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be 'no', 'constant', 'linear', or 'bilinear'.")
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")
    if not ((isinstance(domain_range, tuple) and len(domain_range) == 2 and dimensionality == 1) or (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range) and dimensionality == 2)):
        raise ValueError("`domain_range` must be a tuple for 1D data or a list of two tuples for 2D data.")

    basis = _setup_bspline_basis(dimensionality, p, domain_range, n_basis)

    # Evaluate basis
    B = basis(evaluation_coords)

    B = B[:,:,0].T
    B = B/np.repeat(LpNorm(2)(basis)[np.newaxis,:], B.shape[0], axis=0)

    # Add optional terms to B
    B = _add_optional_terms_to_basis_unstructured(evaluation_coords, B, add_optional_terms, dimensionality, domain_range)

    return B


def _make_prior_dict_basis_builder_fourier_unstructured(evaluation_coords, dimensionality=1, add_optional_terms='no', n_basis=10, domain_range=(0, 1)):
    """
    Constructs and returns a basis matrix for Fourier basis functions, incorporating optional terms and allowing
    unstructured evaluation locations through the parameter evaluation_coords.

    This function facilitates the creation of a basis matrix B, pivotal for modeling in nonparametric functional
    data analysis, especially for periodic data. It supports the addition of 'constant' and 'linear' terms to the
    basis functions, enhancing the model's flexibility and capability to capture underlying data structures across
    one-dimensional and two-dimensional spaces.

    Parameters
    ----------
    evaluation_coords: numpy.array
        A numpy array of shape (n_points, ) in the 1D case or (n_points, 2) in the 2D case
    dimensionality : int, optional
        The dimensionality of the data (1 or 2). Defaults to 1.
        If dimensionality is 2 and any parameter is provided as a single value instead of a tuple or list,
        the single value will be automatically expanded to apply uniformly across both dimensions.
    add_optional_terms : str, optional
        Indicates whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    n_basis : int, optional
        The number of basis functions to use. Defaults to 10.
    domain_range : tuple, optional
        Specifies the domain range over which the basis functions are defined. For 1D data, it is a tuple of
        the form (start, end). For 2D data, a list of such tuples. Defaults to (0, 1).

    Returns
    -------
    numpy.ndarray
        The evaluated and optionally modified basis matrix F, suitable for further analysis or model fitting.

    Raises
    ------
    ValueError
        Raised if any of the input parameters do not conform to their expected formats or allowable values.

    Notes
    -----
    The function seamlessly integrates various steps necessary for preparing the basis matrix for Fourier basis
    functions, including the setup of the basis according to specified parameters, evaluation of the basis on a
    determined grid, and the optional addition of terms to the basis for enhanced modeling capabilities.

    """
    # Validation checks (Examples, additional checks may be required based on function implementation)
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")

    # Validation and expansion for `n_basis`
    if isinstance(n_basis, int):
        n_basis = (n_basis, n_basis) if dimensionality == 2 else n_basis
    elif dimensionality == 2 and (not isinstance(n_basis, tuple) or len(n_basis) != 2):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")

    # Validation for `domain_range`
    if dimensionality == 1:
        if not isinstance(domain_range, tuple):
            raise ValueError("`domain_range` must be a tuple for 1D data.")
    else:  # dimensionality == 2
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Expand to list of tuples for 2D
        elif not (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) for dr in domain_range)):
            raise ValueError("`domain_range` must be a tuple or a list of two tuples for 2D data.")

    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be 'no', 'constant', 'linear', or 'bilinear'.")
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")
    if not ((isinstance(domain_range, tuple) and len(domain_range) == 2 and dimensionality == 1) or (isinstance(domain_range, list) and len(domain_range) == 2 and all(isinstance(dr, tuple) and len(dr) == 2 for dr in domain_range) and dimensionality == 2)):
        raise ValueError("`domain_range` must be a tuple for 1D data or a list of two tuples for 2D data.")


    # Setup Fourier basis
    B = _setup_fourier_basis(dimensionality, n_basis, domain_range)

    # Evaluate basis
    B = B(evaluation_coords)

    B = B[1:,:,0].T

    # Optionally modify the basis matrix
    B = _add_optional_terms_to_basis_unstructured(evaluation_coords, B, add_optional_terms, dimensionality, domain_range)

    return B



def _make_prior_dict_basis_builder_radial_bump_unstructured(evaluation_coords, dimensionality=1, add_optional_terms='no', R=1.0, n_basis=10, domain_range=(0, 1)):
    """
    Constructs and returns a basis matrix for radial basis function over unstructured data, specifically using a bump
    function defined by exp(-1/(1-(r/R)^2)), with the inclusion of optional terms.

    Parameters
    ----------
    evaluation_coords : numpy.array
        Unstructured evaluation coordinates. Shape should be (n_points, ) for 1D and (n_points, 2) for 2D.
    dimensionality : int, optional
        Specifies the dimensionality of the data (1 or 2), defaulting to 1.
    add_optional_terms : str, optional
        Indicates whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions. Options are
        'no' (default), 'constant', 'linear', or 'bilinear'.
    R : float, or tuple of floats
        The radius of the bump radial function. Must be positive. If a tuple is provided, it applies to 2D data.
    n_basis : int, optional
        The number of basis functions to use, defaulting to 10.

    Returns
    -------
    numpy.ndarray
        The evaluated and optionally modified basis matrix B, suitable for further analysis or model fitting.
    """
    # Validation checks
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be either 1 or 2.")
    if add_optional_terms not in ['no', 'constant', 'linear', 'bilinear']:
        raise ValueError("`add_optional_terms` must be one of 'no', 'constant', 'linear', 'bilinear'.")
    if not isinstance(R, (float, tuple)):
        raise ValueError("`R` must be a float or a tuple of floats.")
    if isinstance(R, float) and R <= 0:
        raise ValueError("`R` must be a positive number.")
    if isinstance(R, tuple) and not all(r > 0 for r in R):
        raise ValueError("Each element of `R` must be a positive number.")

    # Ensure R is a tuple for 2D data
    if dimensionality == 2 and isinstance(R, float):
        R = (R, R)

    # Handling single inputs for 2D cases
    if dimensionality == 2:
        if isinstance(domain_range, tuple):
            domain_range = [domain_range, domain_range]  # Duplicate tuple for 2D
        if isinstance(n_basis, int):
            n_basis = (n_basis, n_basis)  # Duplicate value for 2D


    # Setup for basis functions
    if dimensionality == 1 and evaluation_coords.ndim == 1:
        evaluation_coords = evaluation_coords[:, np.newaxis]

    # Validate n_basis format
    if not ((isinstance(n_basis, int) and dimensionality == 1) or (isinstance(n_basis, tuple) and len(n_basis) == 2 and all(isinstance(nb, int) for nb in n_basis) and dimensionality == 2)):
        raise ValueError("`n_basis` must be an integer for 1D data or a tuple of two integers for 2D data.")


    centers = _determine_evaluation_grid(domain_range, None, n_basis, dimensionality)
    if dimensionality == 1:
        centers = centers[:,np.newaxis]
    else:
        centers = np.array(centers)
        centers = np.reshape(centers, (2, np.prod(centers.shape[1:])))
        centers = centers.T

    # Compute distances and evaluate the basis function
    distances = evaluation_coords[:, np.newaxis, :] - centers[np.newaxis, :, :]
    if dimensionality == 2:
        distances[:,:,0] = distances[:,:,0]/R[0]
        distances[:,:,1] = distances[:,:,1]/R[1]
    else:
        distances = distances / R

    distances = np.square(distances).sum(axis=2)

    # Apply the radial bump function
    with np.errstate(all='ignore'):
        B = np.where(distances >= 1.0, 0, np.exp(-1/(1 - distances)))

    grid_points = _determine_evaluation_grid(domain_range, None, 100, dimensionality)


    if dimensionality == 1:
        delta = grid_points[1] - grid_points[0]
        B = B/np.repeat( np.sqrt( np.square(B).sum(axis=0)*delta )[np.newaxis,:] , B.shape[0], axis=0)
    else:
        delta_x = np.diff( grid_points[0][[0,1],0] )[0]
        delta_y = np.diff( grid_points[1][0,[0,1]] )[0]
        B = B/np.repeat( np.sqrt( np.square(B).sum(axis=0)*(delta_x*delta_y) )[np.newaxis,:] , B.shape[0], axis=0)

    B = _add_optional_terms_to_basis_unstructured(evaluation_coords, B, add_optional_terms, dimensionality, domain_range)

    return B





def make_basis_dict_structured(basis_type='bspline', dimensionality=1, p=4, n_basis = 10, R=1.0, domain_range=(0, 1), grid_spacing=None, num_points=10, add_optional_terms='no'):
    """
    Constructs the basis matrix, which is built using either B-spline or Fourier or radial bump basis functions.
    Use `plot_basis_column` to explore the shape of the considered basis functoins.

    Parameters
    ----------
    basis_type : str, optional
        Specifies the type of basis to use: 'bspline' for B-spline basis functions, 'fourier' for Fourier
        basis functions, or 'radial_bump' for radial bump functions. Defaults to 'bspline'.
    dimensionality : int, optional
        The dimensionality of the data (1 or 2), indicating whether the plot should be generated for one-dimensional
        or two-dimensional data. Defaults to 1.
    p : int, optional
        The degree of the B-spline (used when `basis_type` is 'bspline'). Defaults to 4.
    n_basis : int, optional
        The number of basis functions to include in the basis matrix. For `basis_type` 'fourier', this parameter
        is analogous to the number of harmonics. Defaults to 10.
    R : float, optional
        The radius of the bump function, used only when `basis_type` is 'radial_bump'. Must be positive. Defaults to 1.0.
    domain_range : tuple or list of tuples, optional
        The domain range(s) over which the basis functions are evaluated. For 1D data, a tuple of (start, end).
        For 2D data, a list of two tuples, each specifying the domain range for one dimension. Defaults to (0, 1).
    grid_spacing : float or tuple of floats, optional
        The spacing between grid points for evaluating the basis functions. Ignored if `num_points` is specified.
        For 2D data, it can be a tuple specifying the spacing for each dimension.
    num_points : int or tuple of ints, optional
        The number of points for the basis function evaluation. If `grid_spacing` is not provided, this parameter
        is used. For 2D data, it can be a tuple specifying the number of points for each dimension. Defaults to 10.
    add_optional_terms : str, optional
        Specifies whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions, enhancing the
        model's adaptability. Defaults to 'no'.

    Returns
    -------
    A dictionary containing all the hypeparameters of the STAN model relating to the basis function:
        L: int, Number of locations of observation
        p: int, Number of basis funcitons
        B: numpy.ndarray of shape (L,p), The basis matrix

    Raises
    ------
    ValueError
        If `basis_type` is not 'bspline' or 'fourier' or 'radial_bump'.
        If `dimensionality` is not 1 or 2.
        If any other parameter values are outside their expected ranges or formats.

    See Also
    --------
    make_basis_dict_unstructured, initialize_hyperparams_dict, make_prior_dict

    """

    # Validation checks
    if basis_type not in ['bspline', 'fourier', 'radial_bump']:
        raise ValueError("`basis_type` must be 'bspline' or 'fourier' or 'radial_bump'.")
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")

    # Build the basis matrix B
    if basis_type == 'bspline':
        B = _make_prior_dict_basis_builder_bspline(dimensionality=dimensionality, add_optional_terms=add_optional_terms, p=p, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'fourier':
        B = _make_prior_dict_basis_builder_fourier(dimensionality=dimensionality, add_optional_terms=add_optional_terms, n_basis=n_basis, domain_range=domain_range, grid_spacing=grid_spacing, num_points=num_points)
    elif basis_type == 'radial_bump':
        B = _make_prior_dict_basis_builder_radial_bump(dimensionality=dimensionality, add_optional_terms=add_optional_terms, R=R, domain_range=domain_range, n_basis=n_basis, grid_spacing=grid_spacing, num_points=num_points)
    else:
        raise ValueError("Invalid basis_type. Choose 'bspline', 'fourier', or 'radial_bump'.")   

    grid_points = _determine_evaluation_grid(domain_range, grid_spacing, num_points, dimensionality)
    if dimensionality == 2:
        aux = np.array(grid_points)
        grid_points = np.reshape( aux , (2, np.prod( aux.shape[1:] ))).T

    return(
        {
            'L': B.shape[0],
            'p': B.shape[1],
            'B': B,
            't': grid_points
        }
    )


def make_basis_dict_unstructured(evaluation_coords, basis_type='bspline', dimensionality=1, p=4, n_basis = 10, R=1.0, domain_range=(0, 1), add_optional_terms='no'):
    """
    Constructs the basis matrix, which is built using either B-spline or Fourier or radial bump basis functions.
    Use `plot_basis_column` to explore the shape of the considered basis functoins.

    Parameters
    ----------
    evaluation_coords : numpy.array
        Unstructured evaluation coordinates. Shape should be (n_points, ) for 1D and (n_points, 2) for 2D.
    basis_type : str, optional
        Specifies the type of basis to use: 'bspline' for B-spline basis functions, 'fourier' for Fourier
        basis functions, or 'radial_bump' for radial bump functions. Defaults to 'bspline'.
    dimensionality : int, optional
        The dimensionality of the data (1 or 2), indicating whether the plot should be generated for one-dimensional
        or two-dimensional data. Defaults to 1.
    p : int, optional
        The degree of the B-spline (used when `basis_type` is 'bspline'). Defaults to 4.
    n_basis : int, optional
        The number of basis functions to include in the basis matrix. For `basis_type` 'fourier', this parameter
        is analogous to the number of harmonics. Defaults to 10.
    R : float, optional
        The radius of the bump function, used only when `basis_type` is 'radial_bump'. Must be positive. Defaults to 1.0.
    domain_range : tuple or list of tuples, optional
        The domain range(s) over which the basis functions are evaluated. For 1D data, a tuple of (start, end).
        For 2D data, a list of two tuples, each specifying the domain range for one dimension. Defaults to (0, 1).
    add_optional_terms : str, optional
        Specifies whether to add 'constant', 'linear', or 'bilinear' terms to the basis functions, enhancing the
        model's adaptability. Defaults to 'no'.

    Returns
    -------
    A dictionary containing all the hypeparameters of the STAN model relating to the basis function:
        L: int, Number of locations of observation
        p: int, Number of basis funcitons
        B: numpy.ndarray of shape (L,p), The basis matrix

    Raises
    ------
    ValueError
        If `basis_type` is not 'bspline' or 'fourier' or 'radial_bump'.
        If `dimensionality` is not 1 or 2.
        If any other parameter values are outside their expected ranges or formats.

    See Also
    --------
    make_basis_dict_structured, initialize_hyperparams_dict, make_prior_dict

    """

    # Validation checks
    if basis_type not in ['bspline', 'fourier', 'radial_bump']:
        raise ValueError("`basis_type` must be 'bspline' or 'fourier' or 'radial_bump'.")
    if dimensionality not in [1, 2]:
        raise ValueError("`dimensionality` must be 1 or 2.")

    # Build the basis matrix B
    if basis_type == 'bspline':
        B = _make_prior_dict_basis_builder_bspline_unstructured(evaluation_coords, dimensionality=dimensionality, add_optional_terms=add_optional_terms, p=p, domain_range=domain_range, n_basis=n_basis)
    elif basis_type == 'fourier':
        B = _make_prior_dict_basis_builder_fourier_unstructured(evaluation_coords, dimensionality=dimensionality, add_optional_terms=add_optional_terms, n_basis=n_basis, domain_range=domain_range)
    elif basis_type == 'radial_bump':
        B = _make_prior_dict_basis_builder_radial_bump_unstructured(evaluation_coords, dimensionality=dimensionality, add_optional_terms=add_optional_terms, R=R, domain_range=domain_range, n_basis=n_basis)
    else:
        raise ValueError("Invalid basis_type. Choose 'bspline', 'fourier', or 'radial_bump'.")   

    return(
        {
            'L': B.shape[0],
            'p': B.shape[1],
            'B': B,
            't': evaluation_coords.copy()
        }
    )


def initialize_hyperparams_dict(k = 4, v = 5, nu = 4, ψ_alpha = 5, ψ_beta = 4, ψ_sigma_1 = 1.0, ψ_sigma_2 = 1.0, θ_alpha = 5, θ_beta = 4, θ_sigma_1 = 1.0, θ_sigma_2 = 1.0, alpha_1 = 2.1, alpha_2 = 2.5):
    """
    Initializes a dictionary containing hyperparameters for a statistical model. The hyperparameters
    specified in this dictionary are used to define the prior distributions for various parameters
    in the model, including but not limited to, scale parameters for noise distributions and shape
    parameters for prior distributions of model coefficients.

    This function is designed to support a complex statistical model involving multiple likelihood
    and prior distributions as detailed in the model specification. The model uses Normal (Gaussian),
    Inverse-Gamma, and Gamma distributions for likelihoods and priors, and incorporates a Multiplicative
    Gamma Process Shrinkage (MGPS) prior for regularization. Hyperparameters such as `k`, `v`, `nu`,
    `ψ_alpha`, `ψ_beta`, and others directly correspond to the parameters of these distributions and
    influence the model's behavior and performance.

    Parameters
    ----------
    k : int, optional
        Number of Latent Factors hyperparameter. Defaults to 4.
    v : int, optional
        Hyperparameter for degrees of freedom of Student-t of the Latent Factors (Λ). Defaults to 5.
    nu : int, optional
        Hyperparameter for degrees of freedom of Student-t of the Regression Coefficients (β). Defaults to 5.
    ψ_alpha : float, optional
        Alpha parameter for the Inverse-Gamma distribution of the ψ parameter, with a default value of 5.
    ψ_beta : float, optional
        Beta parameter for the Inverse-Gamma distribution of the ψ parameter, with a default value of 4.
    ψ_sigma_1 : float, optional
        Scale multiplier for the first ψ noise parameter, with a default value of 1.0.
    ψ_sigma_2 : float, optional
        Scale multiplier for the second ψ noise parameter, with a default value of 1.0.
    θ_alpha : float, optional
        Alpha parameter for the Gamma distribution of the θ parameter, with a default value of 5.
    θ_beta : float, optional
        Beta parameter for the Gamma distribution of the θ parameter, with a default value of 4.
    θ_sigma_1 : float, optional
        Scale multiplier for the first θ noise parameter, with a default value of 1.0.
    θ_sigma_2 : float, optional
        Scale multiplier for the second θ noise parameter, with a default value of 1.0.
    alpha_1 : float, optional
        Specifies the alpha1 parameter for the Multiplicative Gamma Process Shrinkage, with a default value of 2.1.
    alpha_2 : float, optional
        Specifies the alpha1 parameter for the Multiplicative Gamma Process Shrinkage, with a default value of 2.5.

    Returns
    -------
    dict
        A dictionary containing the initialized hyperparameters for the model. The keys of the dictionary
        correspond to the parameter names (k, v, nu, ψ_alpha, ψ_beta, ψ_sigma_1, ψ_sigma_2, θ_alpha,
        θ_beta, θ_sigma_1, θ_sigma_2, alpha_1, alpha_2), and the values are the initialized values provided
        through the function arguments.

    Examples
    --------
    >>> initialize_hyperparams_dict()
    {
        'k': 4, 'v': 5, 'nu': 4, 'alpha_psi': 5, 'beta_psi': 4, 'psi_noise_scale_multiplier_1': 1.0,
        'psi_noise_scale_multiplier_2': 1.0, 'alpha_sigma': 5, 'beta_sigma': 4, 'theta_noise_scale_multiplier_1': 1.0,
        'theta_noise_scale_multiplier_2': 1.0, 'alpha_1': 2.1, 'alpha_2': 2.5
    }

    Model Specification:
    --------------------
    The statistical model for which these hyperparameters are defined involves complex interactions between
    multiple variables, including observations (Y), basis coefficients (θ), latent variables (η) and
    regression coefficients (β). 
    The likelihood and prior distributions outlined in the model specification guide the formulation of these
    interactions, with specific focus on accommodating the nuances of data distribution and model assumptions.

    The MGPS prior plays a crucial role in regularizing the latent factors and regression coefficients, aiding
    in the prevention of overfitting and ensuring the robustness of the model. The hyperparameters `alpha_1` and
    `alpha_2` are particularly significant for tuning the MGPS prior, influencing the extent of shrinkage applied
    to the latent factors.

    Detailed Model Specification:
    -----------------------------
    The model specifies that observations from two groups (Y^(1) and Y^(2)) are modeled as being independently
    and identically distributed (i.i.d.) from Normal distributions, with means determined by the product of basis
    matrices B^(1), B^(2) and coefficient vectors θ^(1), θ^(2), and variances ψ^(1)^2, ψ^(2)^2, respectively.

    Coefficient vectors θ^(1) and θ^(2) are also modeled as i.i.d. from Normal distributions, with means linked
    to the latent variable η through matrices Λ^(1), Λ^(2) and precision (inverse of variance) parameters τ_θ^(1),
    τ_θ^(2). The latent variable η is, in turn, distributed according to a Normal distribution with mean determined
    by the regression of predictors X on coefficients β.

    The priors for coefficients β, matrices Λ^(1), Λ^(2), and precision parameters τ are defined using a Student-t
    distribution for β and Λ^(1), Λ^(2), a Multiplicative Gamma Process Shrinkage (MGPS) prior for τ_λ^(1), τ_λ^(2),
    and Gamma distributions for τ_θ^(1), τ_θ^(2). The variance parameters ψ^(1)^2, ψ^(2)^2 follow Inverse-Gamma
    distributions.

    This comprehensive modeling approach allows for the flexible adjustment of the model to the specificities of
    the data, guided by the hyperparameters initialized by this function.

    To improve the formulation of the model for ease of sampling in STAN the β_coefficients of the Gamma and
    Inverse-Gamma prior of ψ^2 and τ_θ, are written in the form ψ_beta*ψ_sigma_1, ψ_beta*ψ_sigma_2, 
    θ_beta*θ_sigma_1, θ_beta*θ_sigma_2. It is best to select ψ_beta = ψ_alpha-1, θ_beta = θ_alpha-1 and actually
    tune the β_coefficients through the scale_multipliers.


                  Λ(1)
                    \
        X --> η --> θ(1) -- B(1) --> Y(1)
               \
                \
        Λ(2) --> θ(2) -- B(2) --> Y(2)
    
    See Also
    --------
    make_basis_dict_unstructured, make_basis_dict_structured, make_prior_dict

    """

    return(
        {
            'k': k,
            'v': v,
            'nu': nu,
            'alpha_psi': ψ_alpha,
            'beta_psi': ψ_beta,
            'psi_noise_scale_multiplier_1': ψ_sigma_1,
            'psi_noise_scale_multiplier_2': ψ_sigma_2,
            'alpha_sigma': θ_alpha,
            'beta_sigma': θ_beta,
            'theta_noise_scale_multiplier_1': θ_sigma_1,
            'theta_noise_scale_multiplier_2': θ_sigma_2,
            'alpha_1': alpha_1,
            'alpha_2': alpha_2
        }
    )


def make_prior_dict(hyperparams_dict, B1_dict, B2_dict, y1 = None, y2 = None, X = None, N = None, r = None, y1_missing = None, y2_missing = None):
    """
    Constructs a dictionary containing all the required elements to call either the prior sampler or posterior sampler
    for the statistical model. This includes hyperparameters, basis matrices, observational data, and design matrices.
    
    Parameters
    ----------
    hyperparams_dict : dict
        A dictionary containing hyperparameters for the model. Required keys include 'k', 'v', 'nu', 'alpha_psi',
        'beta_psi', 'psi_noise_scale_multiplier_1', 'psi_noise_scale_multiplier_2', 'alpha_sigma', 'beta_sigma',
        'theta_noise_scale_multiplier_1', 'theta_noise_scale_multiplier_2', 'alpha_1', and 'alpha_2'.
    B1_dict : dict
        A dictionary containing the basis matrix and associated parameters for the first observational dataset.
        Required keys are 'L' (number of locations), 'p' (number of basis functions), 'B' (the basis matrix itself),
        and 't' (evaluation points or times).
    B2_dict : dict
        Similar to `B1_dict`, but for the second observational dataset.
    y1 : numpy.ndarray, optional
        Observational data matrix for the first dataset. Shape should be (L1, N), where L1 is the number of locations
        as specified in `B1_dict` and N is the number of observations. Required if `N` and `r` are not provided.
    y2 : numpy.ndarray, optional
        Observational data matrix for the second dataset. Shape should be (L2, N), where L2 is the number of locations
        as specified in `B2_dict`. Required if `N` and `r` are not provided.
    X : numpy.ndarray, optional
        Design matrix of shape (r, N), where r is the number of regressors. Required if `N` and `r` are not provided.
    N : int, optional
        The number of observations. Required if `y1`, `y2`, and `X` are not provided.
    r : int, optional
        The number of regressors in the design matrix. Required if `y1`, `y2`, and `X` are not provided.
    y1_missing : numpy.ndarray, optional
        A binary matrix indicating missing values in `y1`. Should have the same shape as `y1`.
    y2_missing : numpy.ndarray, optional
        A binary matrix indicating missing values in `y2`. Should have the same shape as `y2`.
    
    Returns
    -------
    dict
        A dictionary containing all the parameters and data necessary for calling the model's sampler functions. This
        includes all inputs plus additional derived quantities such as the number of observations (N), the number of
        locations in each dataset (L1, L2), the number of basis functions in each dataset (p1, p2), and the observational
        data matrices (`y1`, `y2`) and design matrix `X` if they were not provided.
    
    Raises
    ------
    ValueError
        If the input dictionaries do not contain the required keys, if the shapes of the input matrices do not align
        with the specified dimensions, or if there is a mismatch in provided and required parameters.
        If ('y1','y2','X') and ('N','r') are both provided.
    
    Notes
    -----
    The function allows for flexibility in specifying observational data and design matrices, supporting both
    scenarios where the data is directly provided or where the dimensions of the data are specified, and matrices
    are initialized to zeros.

    See Also
    --------
    make_basis_dict_unstructured, make_basis_dict_structured, initialize_hyperparams_dict

    """

    dimensionality1 = None
    dimensionality2 = None

    if not isinstance(hyperparams_dict, dict):
        raise ValueError("hyperparams_dict is not a dictionary.")
    for key in ['k','v','nu','alpha_psi','beta_psi','psi_noise_scale_multiplier_1','psi_noise_scale_multiplier_2','alpha_sigma','beta_sigma','theta_noise_scale_multiplier_1','theta_noise_scale_multiplier_2','alpha_1','alpha_2']:
        if key not in hyperparams_dict:
            raise ValueError(f"hyperparams_dict is missing key {key}")

    # Check if B1_dict is a dictionary
    if not isinstance(B1_dict, dict):
        raise ValueError("B1_dict is not a dictionary.")
    for key in ['L','p','B','t']:
        if key not in B1_dict:
            raise ValueError(f"B1_dict is missing key {key}")
    
    # Check if 'L' is an int or numpy.int_
    if not (isinstance(B1_dict.get('L'), (int, np.int_))):
        raise ValueError("'L' is not an int or numpy.int_.")

    # Check if 'p' is an int or numpy.int_
    if not (isinstance(B1_dict.get('p'), (int, np.int_))):
        raise ValueError("'p' is not an int or numpy.int_.")

    # Check if 'B' is a numpy.array of shape (L, p)
    B1 = B1_dict.get('B')
    L1 = B1_dict['L']
    p1 = B1_dict['p']
    if not (isinstance(B1, np.ndarray) and B1.shape == (B1_dict['L'], B1_dict['p'])):
        raise ValueError("'B1' is not a numpy.array of shape (L1, p1).")

    # Check if 't' is a numpy.array of shape (L, ) or (L, 2)
    t = B1_dict.get('t')
    if not isinstance(t, np.ndarray):
        raise ValueError("'t' is not a numpy.array.")
    
    if t.shape == (B1_dict['L'],):
        dimensionality1 = 1
    elif t.shape == (B1_dict['L'], 2):
        dimensionality1 = 2
    else:
        raise ValueError("'t' is not of shape (L,) or (L,2).")
    t1 = t

    # Check if B2_dict is a dictionary
    if not isinstance(B2_dict, dict):
        raise ValueError("B2_dict is not a dictionary.")
    for key in ['L','p','B','t']:
        if key not in B2_dict:
            raise ValueError(f"B2_dict is missing key {key}")
    
    # Check if 'L' is an int or numpy.int_
    if not (isinstance(B2_dict.get('L'), (int, np.int_))):
        raise ValueError("'L' is not an int or numpy.int_.")

    # Check if 'p' is an int or numpy.int_
    if not (isinstance(B2_dict.get('p'), (int, np.int_))):
        raise ValueError("'p' is not an int or numpy.int_.")

    # Check if 'B' is a numpy.array of shape (L, p)
    B2 = B2_dict.get('B')
    L2 = B2_dict['L']
    p2 = B2_dict['p']
    if not (isinstance(B2, np.ndarray) and B2.shape == (B2_dict['L'], B2_dict['p'])):
        raise ValueError("'B2' is not a numpy.array of shape (L2, p2).")

    # Check if 't' is a numpy.array of shape (L, ) or (L, 2)
    t = B2_dict.get('t')
    if not isinstance(t, np.ndarray):
        raise ValueError("'t' is not a numpy.array.")
    
    if t.shape == (B2_dict['L'],):
        dimensionality2 = 1
    elif t.shape == (B2_dict['L'], 2):
        dimensionality2 = 2
    else:
        raise ValueError("'t' is not of shape (L,) or (L,2).")
    t2 = t

    # Additional checks for y1, y2, X, N, and r
    if y1 is not None and y2 is not None and X is not None:
        if N is not None and r is not None:
            raise ValueError("Since y1, y2 and X where provided, N and r should be None")
        if not (isinstance(y1, np.ndarray) and y1.shape[0] == B1_dict['L']):
            raise ValueError("y1 is not a numpy array of shape (L1, N).")
        if not (isinstance(y2, np.ndarray) and y2.shape[0] == B2_dict['L']):
            raise ValueError("y2 is not a numpy array of shape (L2, N).")
        if not (isinstance(X, np.ndarray)):
            raise ValueError("X is not a numpy array of shape (r, N).")
        N = y1.shape[1]
        r = X.shape[0]
        if not y2.shape[1] == N:
            raise ValueError("y2 is not a numpy array of shape (L2, N).")
        if not X.shape[1] == N:
            raise ValueError("X is not a numpy array of shape (r, N).")
    elif N is not None and r is not None:
        if not isinstance(N, int) or not isinstance(r, int):
            raise ValueError("N and r must be integers.")
        y1 = np.zeros((B1_dict['L'], N))
        y2 = np.zeros((B2_dict['L'], N))
        X = np.zeros((r, N))
    else:
        raise ValueError("Either y1, y2, and X or N and r must be provided.")

    out_dict = hyperparams_dict.copy()
    out_dict['N'] = N
    out_dict['L1'] = L1
    out_dict['L2'] = L2
    out_dict['p1'] = p1
    out_dict['p2'] = p2
    out_dict['t1'] = t1
    out_dict['t2'] = t2
    out_dict['B1'] = B1
    out_dict['B2'] = B2
    out_dict['r'] = r
    out_dict['y1'] = y1
    out_dict['y2'] = y2
    out_dict['X'] = X

    if y1_missing is not None or y2_missing is not None:
        if not (isinstance(y1_missing, np.ndarray) and y1_missing.shape == y1.shape):
            raise ValueError("Mismatch between shape of y1_missing and y1.")
        if not (isinstance(y2_missing, np.ndarray) and y2_missing.shape == y2.shape):
            raise ValueError("Mismatch between shape of y2_missing and y2.")
        out_dict['y1_missing'] = y1_missing.flatten(order='F')
        out_dict['y2_missing'] = y2_missing.flatten(order='F')

    return( out_dict )



