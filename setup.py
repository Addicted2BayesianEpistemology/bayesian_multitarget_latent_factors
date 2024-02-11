from setuptools import setup, find_packages

setup(
    name='bayesian_multitarget_latent_factors',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        # If any package contains *.stan files, include them:
        '': ['*.stan'],
    },
    install_requires=[
        'numpy>=1.26.4',
        'xarray>=2024.1.1',
        'scipy>=1.12.0',
        'xarray_einstats>=0.7.0',
        'ipywidgets>=8.1.2',
        'IPython>=8.21.0',
        'pandas>=2.2.0',
        'matplotlib>=3.8.2',
        'seaborn>=0.13.2',
        'arviz>=0.17.0',
        'scikit-fda>=0.9',
        'cmdstanpy>=1.2.1'
    ],
    include_package_data=True,
    author='Bruno Ursino',
    author_email='bru1uno2@gmail.com',
    description='Fits and explores the posterior of a Bayesian Latent Factor model with multiple functional targets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Addicted2BayesianEpistemology/bayesian_multitarget_latent_factors',
)
