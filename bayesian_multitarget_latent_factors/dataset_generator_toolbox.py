

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

    

