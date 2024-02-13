

import matplotlib as _mpl
import matplotlib.pyplot as _plt
import seaborn as _sns
import pandas as _pd
import xarray as _xr
import numpy as _np
import bayesian_multitarget_latent_factors as bmlf



default_color_dict = {"Tyrian purple":"5f0f40","Carmine":"9a031e",
		              "UT orange":"fb8b24","Spanish orange":"e36414",
		              "Midnight green":"0f4c5c","Olivine":"8cb369",
		              "Flax":"f4e285","Sandy brown":"f4a259",
		              "Viridian":"5b8e7d","Bittersweet shimmer":"bc4b51"}

default_color_dict = \
{
    key: '#'+default_color_dict[key]
    for key in default_color_dict
}


def show_palette(color_dict = None):
    if color_dict is None:
        color_dict_internal = default_color_dict
    else:
        color_dict_internal = color_dict

    # Calculate the number of rows needed (each row contains up to 5 colors)
    num_rows = _np.ceil(len(color_dict_internal) / 5)

    # Adjust the figure size based on the number of rows
    fig, ax = _plt.subplots(figsize=(15, 2 * num_rows))
    
    for i, (name, hex) in enumerate(color_dict_internal.items()):
        # Calculate row and column for current color
        row = i // 5
        col = i % 5
        
        # Position of the rectangle and text are adjusted based on the row and column
        rectangle = _plt.Rectangle((col, row), 1, 1, color=hex)
        ax.add_patch(rectangle)
        
        # Adjust text position and color based on the background brightness
        text_color = _mpl.colors.to_hex(
            _np.array([1, 1, 1]) - _np.array([1, 1, 1]) * (_mpl.colors.rgb_to_hsv(_mpl.colors.hex2color(hex))[2] > 0.5)
        )
        ax.text(col + 0.5, row + 0.5, name, va='center', ha='center', fontsize=10, color=text_color)
    
    # Set limits and turn off axis
    ax.set_xlim(0, 5)  # Fixed to 5 columns
    ax.set_ylim(0, num_rows)  # Dynamic based on number of rows
    ax.axis('off')



def plot_Y(
	ax, idata, test_xr,rng_seed = 200,
	sample_idx = 0, target = 2,
	required = 'predictive', scatter = True,
	bootstrap = None, conditional = False,
	color_dict = None, scatter_color = 'Bittersweet shimmer', errorbar_color = 'Viridian'
	):

    if color_dict is None:
        color_dict_internal = default_color_dict
    else:
        color_dict_internal = color_dict

    rng = _np.random.default_rng(rng_seed)
    
    if not target in [1,2]:
        raise ValueError("Hey!")
    
    from copy import deepcopy

    if target == 1:
        t = idata.constant_data['t1']
    else:
        t = idata.constant_data['t2']

    if not conditional:
        aux_xr = bmlf.sample_unconditional_predictive(
            idata,
            bmlf.renaming_convention( test_xr['X_test']).sel(sample_idx=[sample_idx]).values,
            rng.integers(1000000),
            bootstrap=bootstrap,
            required=required
        )[f'Y{target}'].sel(sample_idx=0)
    else:
        if target == 2:
            aux_xr = bmlf.sample_conditional_predictive(
                idata,
                bmlf.renaming_convention( test_xr['X_test']).sel(sample_idx=[sample_idx]).values,
                rng.integers(1000000),
                bootstrap=bootstrap,
                Y1_test = bmlf.renaming_convention( test_xr['y1']).sel(sample_idx=[sample_idx]).values,
                required=required
            ).sel(sample_idx=0)
        else:
            aux_xr = bmlf.sample_conditional_predictive(
                idata,
                bmlf.renaming_convention( test_xr['X_test']).sel(sample_idx=[sample_idx]).values,
                rng.integers(1000000),
                bootstrap=bootstrap,
                Y2_test = bmlf.renaming_convention( test_xr['y2']).sel(sample_idx=[sample_idx]).values,
                required=required
            ).sel(sample_idx=0)
    
    if bootstrap is None:
        aux_xr = aux_xr.stack(sample = ('chain','draw')).T
    
    _sns.lineplot(
        _pd.DataFrame(
            aux_xr.values,
            columns=t
        ).melt(),
        x='variable',
        y='value',
        errorbar=('pi',90),
        color=color_dict_internal[errorbar_color],
        ax = ax
    )
    
    ax.scatter(t, bmlf.renaming_convention( test_xr[f'y{target}'] ).sel(sample_idx=sample_idx).values, 50, color_dict_internal[scatter_color], )



def plot_Y_wrapper(rng_seed, idata, test_xr, same_sample = True, 
                   sample_ul = 0, sample_ur = 0, sample_ll = 0, sample_lr = 0, 
                   target_ul = 1, target_ur = 2, target_ll = 1, target_lr = 2,
                   req_ul = 'predictive', req_ur = 'predictive', req_ll = 'predictive', req_lr = 'predictive',
                   conditional_ul = False, conditional_ur = False, conditional_ll = True, conditional_lr = True,
                   sharey = 'col', figheight = 11, figwidth = 20
                  ):
    rng = _np.random.default_rng(rng_seed)
    
    fig, axs = _plt.subplots(2,2,sharey=sharey)
    
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)

    if same_sample:
        sample_ul_internal = sample_ul
        sample_ur_internal = sample_ul
        sample_ll_internal = sample_ul
        sample_lr_internal = sample_ul
    else:
        sample_ul_internal = sample_ul
        sample_ur_internal = sample_ur
        sample_ll_internal = sample_ll
        sample_lr_internal = sample_lr
    
    plot_Y(axs[0,0], idata, test_xr, bootstrap=500, rng_seed=rng.integers(1000000),
           sample_idx = sample_ul_internal, target = target_ul,
           required=req_ul, conditional=conditional_ul
          )
    
    plot_Y(axs[0,1], idata, test_xr, bootstrap=500, rng_seed=rng.integers(1000000),
           sample_idx = sample_ur_internal, target = target_ur,
           required=req_ur, conditional=conditional_ur
          )
    
    plot_Y(axs[1,0], idata, test_xr, bootstrap=500, rng_seed=rng.integers(1000000),
           sample_idx = sample_ll_internal, target = target_ll,
           required=req_ll, conditional=conditional_ll
          )
    
    plot_Y(axs[1,1], idata, test_xr, bootstrap=500, rng_seed=rng.integers(1000000),
           sample_idx = sample_lr_internal, target = target_lr,
           required=req_lr, conditional=conditional_lr
          )



def plot_heatmap_regression(idata, pointwise = False, known_target = 1):
    fig, ax = _plt.subplots(1,1)
    if known_target == 1:
        unknown_target = 2
    else:
        unknown_target = 1
    fig.set_figwidth(9)
    fig.set_figheight(5)
    bmlf.get_relationship_between_targets(
        idata, known_target=known_target, pointwise=pointwise
    ).mean(dim='chain').mean(dim='draw').transpose(f'target_{known_target}_dim_idx',f'target_{unknown_target}_dim_idx').plot(ax=ax)





def plot_regression_by_PCA(idata, n_components = 3, pointwise = False, known_target = 1, color_dict = None, random_row_color = True, errorbar_color = 'Spanish orange', pca_component_color = 'Tyrian purple', figwidth = 16, figheight=13):
    
    from sklearn.decomposition import PCA

    rng = _np.random.default_rng(175)

    if color_dict is None:
        color_dict_internal = default_color_dict
    else:
        color_dict_internal = color_dict

    permuted_color_names = rng.permutation(_np.array( [color_name for color_name in color_dict_internal] ))
    
    if known_target == 1:
        unknown_target = 2
    else:
        unknown_target = 1
        
    aux_xr = \
    bmlf.get_relationship_between_targets(
        idata, known_target=known_target, pointwise=pointwise
    ).stack(sample_PCA=('chain','draw',f'target_{known_target}_dim_idx'))
    
    aux = aux_xr.T.values.copy()
    
    pca = PCA(n_components=n_components)
    aux = pca.fit_transform(aux)
    
    PCA_res_xr = \
    _xr.DataArray(
        aux,
        dims = ['sample_PCA','PC_idx'],
        coords = {'sample_PCA':aux_xr.sample_PCA}
    ).unstack('sample_PCA').stack(sample = ('chain','draw')).transpose(f'target_{known_target}_dim_idx','PC_idx','sample')
    
    
    t_unknown = idata.constant_data[f't{unknown_target}']
    t_known = idata.constant_data[f't{known_target}']
    
    fig, axs = _plt.subplots(n_components, 2)
    
    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    fig.suptitle(f'Explained variance by Principal Components: {_np.cumsum(pca.explained_variance_ratio_)}')
    fig.tight_layout()
    
    for i in range(n_components):
        if random_row_color:
            color_component = color_dict_internal[permuted_color_names[i % len(permuted_color_names)]]
            color_scores = color_dict_internal[permuted_color_names[i % len(permuted_color_names)]]
        else:
            color_component = color_dict_internal[pca_component_color]
            color_scores = color_dict_internal[errorbar_color]
        axs[i,1].plot(t_unknown, pca.components_[i], c=color_component)
        axs[i,1].plot(t_unknown, 0.0*t_unknown, 'k--')
        _sns.lineplot(
            _pd.DataFrame(
                PCA_res_xr.sel(PC_idx=i).T.values,
                columns=t_known
            ).melt(),
            x='variable',
            y='value',
            ax=axs[i,0],
            errorbar=('pi',90),
            color=color_scores
        )
        axs[i,0].plot(t_known, 0.0*t_known, 'k--')


def plot_varimaxed_latent_factors(idata, Varimax_res_xr, color_dict = None, random_row_color = True, color_left = 'Spanish orange', color_right = 'Tyrian purple', figwidth = 16, figheight = 15):

    rng = _np.random.default_rng(175)

    if color_dict is None:
        color_dict_internal = default_color_dict
    else:
        color_dict_internal = color_dict

    permuted_color_names = rng.permutation(_np.array( [color_name for color_name in color_dict_internal] ))

    fig, axs = _plt.subplots(idata.constant_data['k'].values[0], 2)

    fig.set_figwidth(figwidth)
    fig.set_figheight(figheight)

    fig.suptitle('Varimaxed Latent Factors')
    fig.tight_layout()

    for k in range(idata.constant_data['k'].values[0]):
        if random_row_color:
            color_left_internal = color_dict_internal[permuted_color_names[k % len(permuted_color_names)]]
            color_right_internal = color_dict_internal[permuted_color_names[k % len(permuted_color_names)]]
        else:
            color_left_internal = color_dict_internal[color_left]
            color_right_internal = color_dict_internal[color_right]
        _sns.lineplot(
            _pd.DataFrame(
                Varimax_res_xr['B1Λ1'].stack(sample=('chain','draw')).sel(latent_factor_idx=k).T.values,
                columns=idata.constant_data['t1']
            ).melt(),
            x='variable',
            y='value',
            ax=axs[k,0],
            errorbar=('pi',90),
            color=color_left_internal
        )
        axs[k,0].plot(idata.constant_data['t1'], 0.0*idata.constant_data['t1'], 'k--')
        _sns.lineplot(
            _pd.DataFrame(
                Varimax_res_xr['B2Λ2'].stack(sample=('chain','draw')).sel(latent_factor_idx=k).T.values,
                columns=idata.constant_data['t2']
            ).melt(),
            x='variable',
            y='value',
            ax=axs[k,1],
            errorbar=('pi',90),
            color=color_right_internal
        )
        axs[k,1].plot(idata.constant_data['t2'], 0.0*idata.constant_data['t2'], 'k--')


def plot_regression_coefficients(Varimax_res_xr, names_array = None, grid = True,
                                 color_dict = None,
                                 red_color = 'Carmine',
                                 green_color = 'Midnight green',
                                 prob_threshold = 0.9,
                                 major_dim = 15,
                                 minor_dim = 8
                                ):

    if color_dict is None:
        color_dict_internal = default_color_dict
    else:
        color_dict_internal = color_dict

    # Adjusting the KDE plots to stack them vertically as in the provided figure
    
    β_xr = Varimax_res_xr['β'].stack(sample=('chain','draw'))
    latent_factors = β_xr.sizes['latent_factor_idx']
    covariates = β_xr.sizes['covariate_idx']
    
    # Create the figure canvas with subplots
    if grid:
        fig, axes = _plt.subplots(covariates , latent_factors, figsize=(major_dim, minor_dim), sharex=True)
    else:
        fig, axes = _plt.subplots(latent_factors * covariates, 1, figsize=(minor_dim, major_dim), sharex=True)
    
    # Flatten the axes array for easy iteration
    #axes_flat = axes.flatten()
    axes_flat = axes.flatten('F')
    if not names_array is None:
        names_array_flat = names_array.reshape((names_array.shape[0]*names_array.shape[1], names_array.shape[2]), order='F')
    
    # Generate the plots
    for i, ax in enumerate(axes_flat):
        latent_factor = i // covariates
        covariate = i % covariates
    
        if _np.prod( _np.quantile( _np.sort( β_xr.sel(latent_factor_idx=latent_factor, covariate_idx=covariate).values.flatten() ) , [(1-prob_threshold)/2, 0.5 + prob_threshold/2] ) ) > 0:
            color = color_dict_internal[green_color]
        else:
            color = color_dict_internal[red_color]
        
        _sns.kdeplot(β_xr.sel(latent_factor_idx=latent_factor, covariate_idx=covariate).values.flatten(), 
                    fill=True, ax=ax, color=color)
    
        
    #    ax.set_title(f'LF {latent_factor}, CV {covariate}', loc='left')
        if β_xr.sel(latent_factor_idx=latent_factor, covariate_idx=covariate).values.flatten().mean() > 0:
            text_xloc = 0.1
            text_halign = 'left'
        else:
            text_xloc = 0.9
            text_halign = 'right'
    
        if names_array is None:
            text = 'Latent Factor = ' + str(latent_factor+1) + '\n   Covariate = ' + str(covariate+1) + '   '
        else:
            text = names_array_flat[i][0] + '\n' + names_array_flat[i][1]
        
        ax.text(text_xloc, .5, text,
                fontweight="medium", color='black', ha=text_halign, va="center", transform=ax.transAxes)
            
        ax.set_ylabel('')  # Remove y-axis label for clarity
        ax.label_outer()  # Hide x-ticks for all but bottom plot
        ax.set_yticks([])
        
        # Remove the spines
        for idx_aux, spine in enumerate(ax.spines.values()):
            if idx_aux != 2:
                spine.set_visible(False)

        if grid:
            ax.axvline(linestyle='--', color='black')

#        if not grid:
#            ax.axvline(linestyle='--')
            
    
    # Set the labels for the bottom plot
    #axes_flat[-1].set_xlabel('β value')
    _plt.tight_layout()

    if not grid:
        overlay_ax = fig.add_axes([0, 0, 1, 1], zorder=1, facecolor='none')
    
        # Disable the overlaying Axes' elements
        overlay_ax.set_axis_off()
    
        x_line_norm = overlay_ax.transAxes.inverted().transform(axes[0].transData.transform((0, 0)))[0]
        y_start_norm = overlay_ax.transAxes.inverted().transform(axes[0].transAxes.transform((0, 1)))[1]
        y_end_norm = overlay_ax.transAxes.inverted().transform(axes[-1].transAxes.transform((0, 0)))[1]
        
        overlay_ax.plot([x_line_norm, x_line_norm], [y_end_norm, y_start_norm], 'k--', transform=overlay_ax.transAxes)
#    overlay_ax.axvline(linestyle='--')
