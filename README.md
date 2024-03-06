
# Bayesian Multitarget Latent Factors

## Introduction
The Bayesian Multitarget Latent Factors package provides a robust solution for multivariate data analysis through Bayesian statistical methods. By leveraging latent factor models, this package aims to uncover hidden structures within datasets that involve multiple targets or outcomes. It is particularly suited for applications where the relationships between variables are complex and intertwined, allowing for a deeper understanding of the underlying factors that drive observable patterns.

## Features

### Latent Factor Modeling
This package specializes in the construction and analysis of Bayesian latent factor models for multivariate data. Latent factor models are instrumental in identifying underlying structures in datasets where the direct relationships between observed variables are not apparent. These models are highly adaptable to a range of applications, from social sciences to bioinformatics, enabling users to uncover hidden factors that influence multiple outcomes or targets simultaneously. By modeling correlations and dependencies, they provide a deeper insight into the complex interactions within the data.

### Varimax Rotation and Post-processing
A key feature of this package is the implementation of Varimax rotation on the posterior samples of the latent factors. Varimax rotation simplifies the interpretation of these factors by maximizing the sum of the variance of the squared loadings. This results in a more orthogonal (uncorrelated) set of factors, making it easier to identify which variables are most strongly associated with each factor. The package includes tools for performing Rotation + Signed Permutation (RSP) to align all samples around the same mode, enhancing the interpretability of the latent structure.

### Advanced Visualization Tools
The package offers a suite of visualization tools designed to facilitate the exploration and presentation of both the data and the results of the analysis. This includes:

- **Heatmaps for Unstructured Data Interpolation**: Utilize `plot_unstructured_heatmap` to generate heatmaps from unstructured data points. This feature is particularly useful for spatial data visualization, where understanding the distribution and gradients of variables across space is crucial.
  
- **3D Surface Plots**: The `plot_3d_with_computed_percentiles` and `plot_3d_with_computed_error_bars` functions allow for the visualization of data and model predictions in three dimensions. These tools can plot 3D surfaces representing the mean values from sample data and include shaded areas or error bars to depict variability, such as confidence intervals or predictive uncertainty.

### Stan Integration for Bayesian Inference
At the core of the package's capabilities is its integration with Stan, a state-of-the-art platform for statistical modeling, Bayesian inference, and predictive modeling. This integration allows for specifying complex latent factor models and performing efficient, scalable inference on these models.

### Extensive Modeling and Analysis Functions
Beyond its primary modeling capabilities, the package includes a variety of functions to support the analysis and interpretation of Bayesian latent factor models. This encompasses utilities for Varimax rotation, functions for projecting test samples into the space of rotated posterior latent factors, and tools for comparing the true latent structure of simulated datasets with the inferred structure. These functions are critical for conducting comprehensive analyses and ensuring the robustness of the findings.

### Posterior Predictive Analysis
The Bayesian Multitarget Latent Factors package extends its functionality to encompass both unconditional and conditional posterior predictive analyses. These features are crucial for understanding the behavior of the model under different conditions and for making predictions based on new data.

- **Unconditional Posterior Predictive Distribution**: This feature allows users to generate predictions from the model without conditioning on new data. It is particularly useful for assessing the model's general behavior and for simulating outcomes based on the distributions learned from the training data. By examining these predictions, users can gain insights into the variability and uncertainty inherent in the model's estimates.

- **Conditional Posterior Predictive Distribution**: In contrast, the conditional posterior predictive functionality enables the projection of new, unseen data into the latent space defined by the model. This approach is invaluable for making predictions about new observations and for evaluating the model's performance on test data. It considers the observed values of new data points to refine these projections, thereby offering a more accurate and tailored predictive performance.

### Advanced Data Projection and Interpretation
Beyond standard modeling and analysis tools, the package offers sophisticated methods for projecting training and test samples onto the varimax-rotated space of posterior latent factors. This capability is essential for:
- Making precise predictions in the rotated latent space.
- Ensuring that interpretations of latent factors remain consistent across different datasets.
- Facilitating the direct comparison between model-based projections and actual observations.

### True Latent Structure Comparison
For studies involving simulated data where the true latent structure is known, the package provides tools to adjust the true latent factors according to the orientation and scaling obtained from the Varimax rotation. This feature allows for a direct comparison between the true latent structure and the structure inferred by the model, offering a powerful method for validating the model's accuracy and for understanding how well the model captures the underlying processes generating the data.

## Installation
### Prerequisites
- CmdStanPy and Stan installation for Bayesian inference

### Steps
1. Ensure you have a Python environment ready. If not, create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```
2. Install the package via pip:
```bash
pip install bayesian-multitarget-latent-factors
```
3. Follow the instructions to install CmdStanPy and Stan as described [here](https://mc-stan.org/cmdstanpy/installation.html#function-install-cmdstan).

## Usage
This package is designed for researchers and data scientists who aim to analyze multivariate datasets through Bayesian latent factor models. Example notebooks are provided in the `examples` directory to demonstrate how to apply these models to real-world datasets, including how to visualize results and interpret latent factors.

## Acknowledgements
- This project utilizes code from the scikit-learn project (https://github.com/scikit-learn/scikit-learn), specifically the _ortho_rotation from decomposition/_factor_analysis, which is licensed under the BSD 3-Clause License. The funciton has been used in the function Varimax_RSP used to interpret the latent factors from the MCMC samples.
- This package builds on the powerful capabilities of Stan for Bayesian modeling.
