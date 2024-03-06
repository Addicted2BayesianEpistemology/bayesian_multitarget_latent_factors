


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


################### Main toolbox for Prior and Posterior sampling ############################

#from .multitarget_latent_factor_model import sample_from_prior, renaming_convention, compute_BLambaBetaX, sample_from_posterior, sample_from_posterior_predictive, sample_conditional_predictive, sample_unconditional_predictive, get_relationship_between_targets, compute_likelihood, compute_likelihood_parallel

from .multitarget_latent_factor_model import sample_from_prior, sample_from_posterior, renaming_convention, compute_likelihood_parallel, get_relationship_between_targets, sample_conditional_predictive, sample_unconditional_predictive, sample_from_posterior_moore_penrose_trick





######################### Varimax rotation toolbox ###########################################

from .varimax_toolbox import Varimax_RSP, sample_projection_on_varimaxed_space, Varimax_true_lambdas





########################## Dictionary Constructor ############################################

#from .dataset_generator_toolbox import dataset_generator, one_functional_target_dictionary_builder, default_basis_dictionary_builder, compactly_supported_radial_basis_fun, compute_euclidean_distances, generate_2D_grid, dataset_generator_2D_domain

from .dataset_generator_toolbox import plot_basis_column, plot_basis_all, make_basis_dict_structured, make_basis_dict_unstructured, initialize_hyperparams_dict, make_prior_dict, plot_col_basis_dict, build_matrix_B_from_gaussian_process_covariance, prior_predictive_properties_from_prior_dict




############################## Plotting toolbox ##############################################

from .plotting_tools import plot_unstructured_heatmap, plot_3d_with_computed_percentiles, plot_3d_with_computed_error_bars, animate_3d_rotation_uncertainty, plot_3_subplots_uncertainty, convert_chain_draw_to_sample, uncertain_lineplot, plot_Y_testing, plot_Y_training, plot_with_credibility_intervals








