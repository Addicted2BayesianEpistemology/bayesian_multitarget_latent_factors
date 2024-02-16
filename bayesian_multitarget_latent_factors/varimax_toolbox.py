
import numpy as np
import arviz as az
import xarray as xr
import xarray_einstats as xrein

from .multitarget_latent_factor_model import renaming_convention

def _Varimax_RSP(Lambda):
    '''
    Lambda should be a numpy array in which the first dimension is the MCMC draw, the second is the basis function and the third is the latent factor
    '''
    from scipy.optimize import linear_sum_assignment
    from itertools import product

    def _ortho_rotation(components, method="varimax", tol=1e-6, max_iter=100):
        """Return rotated components."""
        nrow, ncol = components.shape
        rotation_matrix = np.eye(ncol)
        var = 0
    
        for _ in range(max_iter):
            comp_rot = np.dot(components, rotation_matrix)
            if method == "varimax":
                tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
            elif method == "quartimax":
                tmp = 0
            u, s, v = np.linalg.svd(np.dot(components.T, comp_rot**3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + tol):
                break
            var = var_new
    
        return np.dot(components, rotation_matrix), rotation_matrix
    
    T = Lambda.shape[0]
    p = Lambda.shape[1]
    q = Lambda.shape[2]
    Λ = np.zeros( Lambda.shape )
    Phi = np.zeros( (T,q,q) )
    for t in range(T):
        Λ[t, :, :], Phi[t,:,:] = _ortho_rotation(Lambda[t, :, :])
        if t % 500 == 0:
            print('Rotated sample ' + str(t))        

    S = np.ones((T, q))
    ν = np.repeat(
        np.arange(q)[np.newaxis, :],
        T,
        axis=0
    )
    
    all_s = np.array( list(product([-1,1], repeat=q)) )

    objective_fun = +np.inf

    iteration_number = 0
    while True:
        print('Starting iteration number ' + str(iteration_number))
        iteration_number += 1
        objective_fun_new = 0
        Λstar = np.zeros((p,q))
        for t in range(T):
            Λaux = (Λ[t,:,:] @ np.diag(S[t,:]))
            for i in range(q):
                Λstar[:,i] += Λaux[:,ν[t,i]]/T
        for t in range(T):
            ν_new = np.zeros( all_s.shape )
            cost = np.zeros( all_s.shape[0] )
            for s_idx in range(all_s.shape[0]):
                s = all_s[s_idx,:]
                Λaux = \
                np.matmul(
                    Λ[t,:,:],
                    np.diag(s)
                )
                C = np.zeros((q,q))
                for j in range(q):
                    C[:,j] = np.square( Λstar[:,:] - Λaux[:,[j]] ).sum(axis=0)
                row_idx, col_idx = linear_sum_assignment(C)
                cost[s_idx] = C[row_idx, col_idx].sum()
                ν_new[s_idx, row_idx] = col_idx
            s_idx_best = np.argmin(cost)
            ν[t,:] = ν_new[s_idx_best, :]
            S[t,:] = all_s[s_idx_best, :]
            objective_fun_new += min(cost)
        if np.isclose(objective_fun_new, objective_fun, rtol=1e-4):
            break
        if iteration_number >= 50:
            break
        print('\t Previous objective fun =\t' + "{:.{}f}".format(objective_fun, 3) + '\n\t New objective fun =\t\t' + "{:.{}f}".format(objective_fun_new, 3))
        objective_fun = objective_fun_new

    for t in range(T):
        Λ[t,:,:] = Λ[t,:,:] @ np.diag(S[t,:])
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
    if not isinstance(idata, az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData object.")

    aux_xr = xr.concat(
        [
            renaming_convention( idata.posterior['Lambda1'] ).rename({'basis_fun_branch_1_idx':'basis_fun_branch_idx'}),
            renaming_convention( idata.posterior['Lambda2'] ).rename({'basis_fun_branch_2_idx':'basis_fun_branch_idx'}),
        ],
        'basis_fun_branch_idx'
    ).stack(chain_draw_idx = ('chain','draw')).transpose('chain_draw_idx','basis_fun_branch_idx','latent_factor_idx')

    Phi, Λ, ν, S = _Varimax_RSP(aux_xr.values)
    
    Phi_xr = \
    xr.DataArray(
        Phi,
        dims=['chain_draw_idx','latent_factor_idx','latent_factor_idx_bis'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx','latent_factor_idx_bis')

    p1 = idata.constant_data['p1'].values[0]
    p2 = idata.constant_data['p2'].values[0]

    Λ1_xr = \
    xr.DataArray(
        Λ[:,0:p1,:],
        dims=['chain_draw_idx','basis_fun_branch_1_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','basis_fun_branch_1_idx','latent_factor_idx')

    Λ2_xr = \
    xr.DataArray(
        Λ[:,p1:(p1+p2),:],
        dims=['chain_draw_idx','basis_fun_branch_2_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','basis_fun_branch_2_idx','latent_factor_idx')

    P = np.zeros((ν.shape[0],ν.shape[1],ν.shape[1]))
    for t in range(ν.shape[0]):
        P[t,np.arange(ν.shape[1]),ν[t]] = 1

    # warning! bis comes first because we are hiding a transposition here
    P_xr = \
    xr.DataArray(
        P,
        dims=['chain_draw_idx','latent_factor_idx_bis','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx','latent_factor_idx_bis')

    S_xr = \
    xr.DataArray(
        S,
        dims=['chain_draw_idx','latent_factor_idx'],
        coords={
            'chain_draw_idx': aux_xr.indexes['chain_draw_idx']
        }
    ).unstack().transpose('chain','draw','latent_factor_idx')

    B1Λ1_xr = \
    xrein.linalg.matmul(
        renaming_convention( idata.constant_data['B1'] ),
        Λ1_xr,
        dims=[['target_1_dim_idx','basis_fun_branch_1_idx'],['basis_fun_branch_1_idx','latent_factor_idx']]
    )    

    B2Λ2_xr = \
    xrein.linalg.matmul(
        renaming_convention( idata.constant_data['B2'] ),
        Λ2_xr,
        dims=[['target_2_dim_idx','basis_fun_branch_2_idx'],['basis_fun_branch_2_idx','latent_factor_idx']]
    )    

    Varimaxed_β_xr = \
    xrein.linalg.matmul(
        P_xr,
        xrein.linalg.matmul(
            Phi_xr,
            renaming_convention( idata.posterior['beta'] ),
            dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','covariate_idx']]
        ).rename({'latent_factor_idx_bis':'latent_factor_idx'})*S_xr,
        dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','covariate_idx']]
    ).rename({'latent_factor_idx_bis':'latent_factor_idx'})    

    return(
        xr.Dataset(
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
    if not isinstance(rng_seed, (int, np.int_)):
        raise ValueError("rng_seed must be an integer.")
    
    # Validate idata
    if not isinstance(idata, az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData object.")
    
    # Validate Varimax_res_xr
    if not isinstance(Varimax_res_xr, xr.Dataset):
        raise ValueError("Varimax_res_xr must be an xarray.Dataset object.")
    
    # Validate X_test, if provided
    if X_test is not None:
        if not isinstance(X_test, np.ndarray) or X_test.ndim != 2:
            raise ValueError("X_test must be a 2D numpy array.")
    
    # Validate Y1_test, if provided
    if Y1_test is not None:
        if not isinstance(Y1_test, np.ndarray) or Y1_test.ndim != 2:
            raise ValueError("Y1_test must be a 2D numpy array.")
    
    # Validate Y2_test, if provided
    if Y2_test is not None:
        if not isinstance(Y2_test, np.ndarray) or Y2_test.ndim != 2:
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

    rng = np.random.default_rng(rng_seed)

    if X_test is None:
        # the order ['latent_factor_idx_bis','latent_factor_idx'] implies a transposition of the rotation matrix
        res_xr = \
        xrein.linalg.matmul(
            Varimax_res_xr['P'],
            xrein.linalg.matmul(
                Varimax_res_xr['R'],
                renaming_convention( idata.posterior['eta'] ),
                dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','sample_idx']]
            ).rename({'latent_factor_idx_bis':'latent_factor_idx'})*Varimax_res_xr['S'],
            dims=[['latent_factor_idx_bis','latent_factor_idx'],['latent_factor_idx','sample_idx']]
        ).rename({'latent_factor_idx_bis':'latent_factor_idx'})
        return( res_xr )

    xr_dataset = deepcopy( idata.posterior )

    η_estimate = \
    xrein.linalg.matmul(
        renaming_convention( xr_dataset['beta'] ),
        xr.DataArray(X_test, dims=['covariate_idx','sample_idx']),
        dims=[['latent_factor_idx','covariate_idx'],['covariate_idx','sample_idx']]
    )

    Σ = xr.DataArray(np.eye(η_estimate.sizes['latent_factor_idx']), dims=['latent_factor_idx','latent_factor_idx2'])
    aux_xr = sample_unconditional_predictive(idata, X_test, 13, required='predictive estimate')

    if (Y1_test is None) and (Y2_test is None):
        n_provided_Ys = 0
    elif (not Y1_test is None) and (not Y2_test is None):
        n_provided_Ys = 2
        Y1_test_xr = xr.DataArray(
            Y1_test,
            dims=['target_1_dim_idx','sample_idx']
        )
        Y2_test_xr = xr.DataArray(
            Y2_test,
            dims=['target_2_dim_idx','sample_idx']
        )
        ΔY_xr = \
        xr.concat(
            [
                renaming_convention( Y1_test_xr ).rename({'target_1_dim_idx':'target_dim_idx'}),
                renaming_convention( Y2_test_xr ).rename({'target_2_dim_idx':'target_dim_idx'})
            ],
            dim='target_dim_idx'
        ) - xr.concat(
            [
                aux_xr['Y1'].rename({'target_1_dim_idx':'target_dim_idx'}),
                aux_xr['Y2'].rename({'target_2_dim_idx':'target_dim_idx'})
            ],
            dim='target_dim_idx'
        )

        invΣy = \
        xrein.linalg.inv(
            xr.concat(
                [
                    xr.concat(
                        [
                            renaming_convention( xr_dataset['Sigma_11'] ).rename({'target_1_dim_idx_bis':'target_dim_idx2'}),
                            renaming_convention( xr_dataset['Sigma_12'] ).rename({'target_2_dim_idx':'target_dim_idx2'})
                        ],
                        dim='target_dim_idx2'
                    ).rename({'target_1_dim_idx':'target_dim_idx'}),
                    xr.concat(
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
        xr.concat(
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
            Y1_test_xr = xr.DataArray(
                Y1_test,
                dims=['target_1_dim_idx','sample_idx']
            )
            g = 1
            ΔY_xr = Y1_test_xr.rename({'target_1_dim_idx':'target_dim_idx'}) - aux_xr['Y1'].rename({'target_1_dim_idx':'target_dim_idx'})
        else:
            # Y2 is provided Y1 not
            Y2_test_xr = xr.DataArray(
                Y2_test,
                dims=['target_2_dim_idx','sample_idx']
            )
            g = 2
            ΔY_xr = Y2_test_xr.rename({'target_2_dim_idx':'target_dim_idx'}) - aux_xr['Y2'].rename({'target_2_dim_idx':'target_dim_idx'})
        invΣy = xrein.linalg.inv(
            renaming_convention( xr_dataset[f'Sigma_{g}{g}'] ).rename({f'target_{g}_dim_idx':'target_dim_idx',f'target_{g}_dim_idx_bis':'target_dim_idx2'}),
            dims=['target_dim_idx','target_dim_idx2']
        )
        Σyη = Varimax_res_xr[f'B{g}Λ{g}'].rename({f'target_{g}_dim_idx':'target_dim_idx'})


    if not ((Y1_test is None) and (Y2_test is None)):
        η_estimate += \
        xrein.linalg.matmul(
            Σyη,
            xrein.linalg.matmul(
                invΣy,
                ΔY_xr,
                dims=[['target_dim_idx','target_dim_idx2'],['target_dim_idx','sample_idx']]
            ),
            dims=[['latent_factor_idx','target_dim_idx'],['target_dim_idx','sample_idx']]
        )

        Σ = \
        xrein.linalg.matmul(
            Σyη,
            xrein.linalg.matmul(
                invΣy,
                Σyη,
                dims=[['target_dim_idx','target_dim_idx2'],['target_dim_idx','latent_factor_idx']]
            ),
            dims=[['latent_factor_idx','target_dim_idx'],['target_dim_idx','latent_factor_idx']]
        ) - Σ
        Σ = -Σ

    CholeskyΣ = xrein.linalg.cholesky(
        Σ,
        dims=['latent_factor_idx','latent_factor_idx2']
    )

    η_predictive = deepcopy(η_estimate)
    η_predictive += \
    xrein.linalg.matmul(
        CholeskyΣ,
        xr.DataArray(
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
