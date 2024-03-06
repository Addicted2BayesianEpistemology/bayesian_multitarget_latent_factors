import numpy as np
import matplotlib.pyplot as plt


__all__ = ['plot_unstructured_heatmap',]


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





def plot_unstructured_heatmap(values, locations, method='cubic', grid_res=100, plot_points=False, ax=None, xlabel=None, ylabel=None, title='Heatmap of Interpolated Values', colorbar_label='Interpolated Value', show_colorbar=True, vmin = None, vmax = None, colormap='viridis', max_distance=None, p_NN_distance_max_distance = np.inf, use_contour = False, **kwargs):
    """
    Generates a heatmap by interpolating the given values at specified locations, using a specified interpolation method. The heatmap is plotted on a grid with a resolution determined by `grid_res`. This function can also optionally plot the original data points on the heatmap.

    The interpolation and visualization are customizable, allowing for adjustments to the interpolation method, grid resolution, color map, and other aspects of the plot. This function is useful for visualizing spatial data, understanding patterns, and identifying regions of interest within a dataset.

    Parameters
    ----------
    values : numpy.ndarray
        A 1D array of values to be interpolated.
    locations : numpy.ndarray
        A 2D array of shape (n, 2), representing the x and y coordinates of each value in `values`.
    method : str, optional
        The method of interpolation to use. Options are 'linear', 'nearest', and 'cubic'. Defaults to 'cubic'.
    grid_res : int or complex, optional
        The resolution of the interpolation grid. If an integer, it specifies the number of points in each dimension.
        If a complex number, its real and imaginary parts specify the number of points along the x and y axes, respectively.
        Defaults to 100.
    plot_points : bool, optional
        If True, the original points are plotted on the heatmap. Defaults to False.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmap. If None, the current axes will be used. Defaults to None.
    xlabel : str, optional
        The label for the x-axis. Defaults to None.
    ylabel : str, optional
        The label for the y-axis. Defaults to None.
    title : str, optional
        The title of the plot. Defaults to 'Heatmap of Interpolated Values'.
    colorbar_label : str, optional
        The label for the colorbar. Applies only if `show_colorbar` is True. Defaults to 'Interpolated Value'.
    show_colorbar : bool, optional
        Whether to display the colorbar. Defaults to True.
    vmin, vmax : float, optional
        The minimum and maximum values used for the colormap. If None, they are inferred from the data. Defaults to None.
    colormap : str, optional
        The colormap for the heatmap. Defaults to 'viridis'.
    max_distance : float, optional
        The maximum distance for interpolation. Points farther than this distance from a grid point will not influence the
        interpolation at that grid point. Defaults to None, meaning there is no limit.
    p_NN_distance_max_distance : float, optional
        The power parameter for the nearest neighbor distance calculation when `max_distance` is specified. Defaults to infinity,
        indicating standard Euclidean distance.
    use_contour : bool, optional
        If True, uses plt.contourf to plot the heatmap instead of plt.imshow. Defaults to False.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the heatmap.
    matplotlib.image.AxesImage
        The image object representing the heatmap.
    matplotlib.colorbar.Colorbar or None
        The colorbar object, if `show_colorbar` is True. Otherwise, None.
    float
        The minimum value of the color scale used in the plot.
    float
        The maximum value of the color scale used in the plot.

    Raises
    ------
    ValueError
        If input parameters are not within their expected ranges or types.

    Examples
    --------
    >>> values = np.random.rand(100)
    >>> locations = np.random.rand(100, 2)
    >>> ax, img, cbar, vmin, vmax = plot_unstructured_heatmap(values, locations, method='cubic', grid_res=100,
                                                              plot_points=True, colormap='viridis')
    """



    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError("values must be a 1D numpy array.")
    if not isinstance(locations, np.ndarray) or locations.shape[1] != 2:
        raise ValueError("locations must be a numpy array of shape (n, 2).")
    if method not in ['linear', 'nearest', 'cubic']:
        raise ValueError("method must be 'linear', 'nearest', or 'cubic'.")
    if not (isinstance(grid_res, int) and grid_res > 0 or isinstance(grid_res, complex)):
        raise ValueError("grid_res must be a positive integer or a complex number.")
    if not isinstance(plot_points, bool):
        raise ValueError("plot_points must be a boolean.")
    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None.")
    for label in [xlabel, ylabel, title, colorbar_label]:
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{label} must be a string or None.")
    if not isinstance(show_colorbar, bool):
        raise ValueError("show_colorbar must be a boolean.")
    if not isinstance(colormap, str):
        raise ValueError("colormap must be a string.")
    if max_distance is not None and not (isinstance(max_distance, (int, float)) and max_distance > 0):
        raise ValueError("max_distance must be a positive number or None.")

    # Define the grid
    grid_x, grid_y = np.mgrid[min(locations[:, 0]):max(locations[:, 0]):complex(grid_res), 
                              min(locations[:, 1]):max(locations[:, 1]):complex(grid_res)]

    # Create a KDTree for efficient distance computation
    tree = cKDTree(locations)
    
    # Interpolate the data
    grid_z = griddata(locations, values, (grid_x, grid_y), method=method)
    
    # Mask out areas too far from the nearest data point if max_distance is specified
    if max_distance is not None:
        # Find distance to nearest point
        distances, _ = tree.query(np.vstack([grid_x.ravel(), grid_y.ravel()]).T, k=1, p = p_NN_distance_max_distance)
        mask = distances.reshape(grid_x.shape) > max_distance
        grid_z[mask] = np.nan  # Use NaN for areas to leave as white space

    # Determine color scale limits if not provided
    if vmin is None or vmax is None:
        valid_values = grid_z[~np.isnan(grid_z)]
        vmin = valid_values.min() if vmin is None else vmin
        vmax = valid_values.max() if vmax is None else vmax

    # Create plot on the specified axes
    if ax is None:
        ax = plt.gca()

    if use_contour:
        img = ax.contourf(grid_x, grid_y, grid_z, levels=np.linspace(vmin, vmax, 10), cmap=colormap, linestyles='solid', **kwargs)
    else:
        img = ax.imshow(grid_z.T, extent=(min(locations[:, 0]), max(locations[:, 0]), min(locations[:, 1]), max(locations[:, 1])), origin='lower', aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)

    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(img, ax=ax, label=colorbar_label)
    if plot_points:
        ax.plot(locations[:, 0], locations[:, 1], 'k.', markersize=5, alpha=0.5)  # plot the original points
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax, img, cbar,vmin, vmax



def plot_3d_with_computed_percentiles(xy, z_samples, percentiles=(5, 95), ax=None, grid_res=100, azimuth_angle = 30):
    """
    Plots a 3D surface of mean values from sample data with shaded areas representing specified percentile bounds.
    This function computes the mean and the lower and upper percentiles for each point in a set of sample data,
    interpolates these values onto a structured grid, and then plots the mean values as a 3D surface.
    Shaded areas around the surface indicate the range between the lower and upper percentiles,
    providing a visual representation of the variability or uncertainty in the data.

    Parameters
    ----------
    xy : numpy.ndarray
        A 2D array of shape (n, 2), representing the x and y coordinates of each sample point in `z_samples`.
    z_samples : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of sample points and m is the number of samples at each point.
        This array contains the z values (e.g., measurements or simulations) for each of the sample points.
    percentiles : tuple of int, optional
        A tuple containing the lower and upper percentiles to compute and display around the mean surface.
        Defaults to (5, 95), which shows the 5th and 95th percentiles.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        The 3D axes on which to plot the surface and percentiles.
        If None, a new figure and 3D axes are created. Defaults to None.
        Can also take non 3D axes, in which automatically corrects to 3D
    grid_res : int, optional
        The resolution of the grid used for interpolating the mean and percentile surfaces. Higher values result in finer grids and more detailed surfaces but require more computation. Defaults to 100.
    azimuth_angle : float, optional
        The azimuth angle in degrees for the initial view of the 3D plot. This angle determines the direction from which the plot is viewed. Defaults to 30.

    Returns
    -------
        fig, ax

    Examples
    --------
    >>> xy = np.random.rand(100, 2)
    >>> z_samples = np.random.rand(100, 10)  # 10 samples for each of the 100 points
    >>> plot_3d_with_computed_percentiles(xy, z_samples, percentiles=(5, 95), grid_res=100, azimuth_angle=30)
        The function plots a 3D surface of the mean values interpolated from the sample data, with shaded regions indicating the 5th and 95th percentiles.
    """
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None or not isinstance(ax, Axes3D):
        if ax is not None:
            fig = ax.figure
            pos = ax.get_position()  # Get the original axes position
            ax.remove()  # Remove the existing axes
            ax = fig.add_subplot(111, projection='3d', position=pos)  # Create a new 3D axes in the same position
        else:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
    
    # Compute mean and percentile bounds
    z_mean = np.mean(z_samples, axis=1)
    z_lower = np.percentile(z_samples, percentiles[0], axis=1)
    z_upper = np.percentile(z_samples, percentiles[1], axis=1)
    
    # Create grid
    x_lin = np.linspace(np.min(xy[:, 0]), np.max(xy[:, 0]), grid_res)
    y_lin = np.linspace(np.min(xy[:, 1]), np.max(xy[:, 1]), grid_res)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    
    # Interpolate mean and percentile bounds onto structured grid
    z_grid = griddata(xy, z_mean, (x_grid, y_grid), method='cubic')
    z_lower_grid = griddata(xy, z_lower, (x_grid, y_grid), method='cubic')
    z_upper_grid = griddata(xy, z_upper, (x_grid, y_grid), method='cubic')
    
    # Plot the main surface
    ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.9)
    
    # Plot softly colored regions for the upper and lower percentiles
    ax.plot_surface(x_grid, y_grid, z_upper_grid, color='red', alpha=0.2)
    ax.plot_surface(x_grid, y_grid, z_lower_grid, color='red', alpha=0.2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=azimuth_angle)

    return fig, ax
    


def plot_3d_with_computed_error_bars(xy, z_samples, percentiles=(5, 95), ax=None, grid_res=100, step=1, azimuth_angle = 30):
    """
    Plots a 3D surface with computed error bars using samples of z-values at given xy locations.
    This function visualizes the mean of the z-values across samples and indicates variability with
    error bars at specified percentiles.

    The 3D surface is interpolated from the mean z-values at the given locations,
    and error bars are plotted for a subset of these points,
    determined by the `step` parameter.
    This visualization is useful for understanding the uncertainty or variability of data across a spatial domain.

    Parameters
    ----------
    xy : numpy.ndarray
        A 2D array of shape (n, 2), representing the x and y coordinates of each sample.
    z_samples : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of samples and m is the number of measurements per sample.
        This array contains the z-values for each sample at the locations specified in `xy`.
    percentiles : tuple, optional
        A tuple of two values indicating the lower and upper percentiles to use for computing error bars.
        Defaults to (5, 95).
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        The 3D axes on which to plot the surface and error bars.
        If None, a new figure and 3D axes are created. Defaults to None.
        Can also take non 3D axes, in which automatically corrects to 3D
    grid_res : int, optional
        The resolution of the grid used for interpolating the surface.
        This value specifies the number of points along each axis of the grid. Defaults to 100.
    step : int, optional
        The step size for selecting points from `xy` at which to plot error bars.
        A smaller step size results in more error bars being plotted. Defaults to 1.
    azimuth_angle : int, optional
        The azimuth angle for the 3D plot's perspective. Defaults to 30.

    Returns
    -------
        fig, ax

    Examples
    --------
    >>> xy = np.random.rand(100, 2)
    >>> z_samples = np.random.rand(100, 10)  # 10 measurements per sample
    >>> plot_3d_with_computed_error_bars(xy, z_samples, percentiles=(5, 95), grid_res=100, step=10, azimuth_angle=30)
    """

    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None or not isinstance(ax, Axes3D):
        if ax is not None:
            fig = ax.figure
            pos = ax.get_position()  # Get the original axes position
            ax.remove()  # Remove the existing axes
            ax = fig.add_subplot(111, projection='3d', position=pos)  # Create a new 3D axes in the same position
        else:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
    
    # Compute mean and percentile bounds
    z_mean = np.mean(z_samples, axis=1)
    z_lower = np.percentile(z_samples, percentiles[0], axis=1)
    z_upper = np.percentile(z_samples, percentiles[1], axis=1)
    
    # Create grid
    x_lin = np.linspace(np.min(xy[:, 0]), np.max(xy[:, 0]), grid_res)
    y_lin = np.linspace(np.min(xy[:, 1]), np.max(xy[:, 1]), grid_res)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    
    # Interpolate mean and percentile bounds onto structured grid
    z_grid = griddata(xy, z_mean, (x_grid, y_grid), method='cubic')
    
    # Plot the main surface
    ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.9)
    
    # Plot error bars for a subset of points
    for i in range(0, xy.shape[0], step):
        x_pt, y_pt = xy[i, :]
#        z_low = griddata(xy, z_lower, (x_pt, y_pt), method='cubic')
#        z_up = griddata(xy, z_upper, (x_pt, y_pt), method='cubic')

        z_low = z_lower[i]
        z_up = z_upper[i]

        # Plotting the error bar for this point
        ax.plot([x_pt, x_pt], [y_pt, y_pt], [z_low, z_up], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(azim=azimuth_angle)

    return fig, ax


def animate_3d_rotation_uncertainty(xy, z_samples, percentiles=(5, 95), grid_res=100, step=1, filename='rotation.gif', error_bars = True, rotation_steps = 72, rot_interval=100):
    """
    Creates an animated GIF showing the rotation of the 3D plot around the z-axis.

    Parameters:
    - xy: Numpy array of shape (n_points, 2) containing x and y coordinates.
    - z_samples: Array of z-values of shape (n_points, n_samples) with multiple samples per point.
    - percentiles: Tuple containing lower and upper percentile bounds to compute.
    - grid_res: Resolution of the grid for interpolation.
    - step: Step size for plotting error bars to avoid overcrowding. (only used with error bars)
    - filename: Name of the output GIF file.
    - error_bars: True to obtain the error bars, False to get the cloud
    - rot_interval: higher is slower (sorry)
    """
    
    from matplotlib.animation import FuncAnimation
    import tempfile
    import os
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    angles = np.linspace(-180, 180, rotation_steps)  # Generate angles from -180 to 180
    
    def update(angle):
        ax.view_init(azim=angle)
        ax.cla()  # Clear the current axes
        if error_bars:
            plot_3d_with_computed_error_bars(xy, z_samples, percentiles, ax=ax, grid_res=grid_res, step=step, azimuth_angle=angle)
        else:
            plot_3d_with_computed_percentiles(xy, z_samples, percentiles, ax=ax, grid_res=grid_res, azimuth_angle=angle)

    anim = FuncAnimation(fig, update, frames=angles, interval=rot_interval)
    
    # Save the animation
    anim.save(filename, writer='imagemagick')



def plot_3_subplots_uncertainty(xy, z_samples, percentiles=(5, 95), ax=None, grid_res=100, vmin = None, vmax = None, colormap = 'coolwarm'):
    """
    Generates a set of three subplots visualizing the uncertainty of interpolated values at specified locations,
    based on samples of those values.
    The subplots include heatmaps for the lower and upper percentiles to show the range of uncertainty,
    and a heatmap of significance indicating areas of strong positive or negative deviation.

    This function is useful for visualizing spatial uncertainty,
    understanding variability across a dataset,
    and highlighting significant regions of interest.

    Parameters
    ----------
    xy : numpy.ndarray
        A 2D array of shape (n, 2), representing the x and y coordinates of each location.
    z_samples : numpy.ndarray
        A 2D array where each row corresponds to a location in `xy` and each column is a sampled value at that location.
    percentiles : tuple of two floats, optional
        The lower and upper percentiles to use for calculating the uncertainty bounds. Defaults to (5, 95).
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmaps. If None, the current axes will be used. Defaults to None.
    grid_res : int or complex, optional
        The resolution of the interpolation grid. Defaults to 100.
    vmin, vmax : float, optional
        The minimum and maximum values used for the colormap of the upper and lower percentile heatmaps.
        If None, they are inferred from the data. Defaults to None.
    colormap : str, optional
        The colormap for the upper and lower percentile heatmaps. Defaults to 'coolwarm'.
    
    Returns
    -------
    float
        The minimum value of the color scale used in the percentile plots.
    float
        The maximum value of the color scale used in the percentile plots.

    Raises
    ------
    ValueError
        If input parameters are not within their expected ranges or types.

    Examples
    --------
    >>> xy = np.random.rand(100, 2)
    >>> z_samples = np.random.randn(100, 1000)
    >>> vmin, vmax = plot_3_subplots_uncertainty(xy, z_samples, percentiles=(5, 95),
                                                  grid_res=100, colormap='coolwarm')
    """
    def _create_subsubplots(ax, texts = ['A','B','C'], scalefactor = '43%'):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Ensure the original axes is turned off to not interfere visually
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Define dimensions for the subsubplots relative to the original axes
        width, height = scalefactor, scalefactor
    
        subaxes = []
        # Create 3 subplots for plotting graphs
        for i in range(3):
            locs = ['upper left','lower left','lower right']
            subax = inset_axes(ax, width=width, height=height,
                               bbox_transform=ax.transAxes,
                               bbox_to_anchor=(0, 0, 1, 1),
                               loc=locs[i] )
    
            # Basic subplot formatting
            subax.spines['top'].set_visible(False)
            subax.spines['right'].set_visible(False)
            subaxes.append(subax)
        
        # Create the 4th subplot for the 2x2 grid of letters
        grid_ax = inset_axes(ax, width=width, height=height,
                             bbox_transform=ax.transAxes,
                             bbox_to_anchor=(0, 0, 1, 1),
                             loc='upper right')
        
        # Hide the axes for the grid
        grid_ax.set_xticks([])
        grid_ax.set_yticks([])
        grid_ax.spines['top'].set_visible(False)
        grid_ax.spines['right'].set_visible(False)
        grid_ax.spines['bottom'].set_visible(False)
        grid_ax.spines['left'].set_visible(False)
        
        # Add letters in a 2x2 grid within the 4th subplot
        letters = [texts[0], '', texts[1], texts[2]]
        for i, letter in enumerate(letters):
            grid_ax.text((i % 2) * 0.5 + 0.25, (1 - i // 2) * 0.5 + 0.25, letter,
                         ha='center', va='center', fontsize=12)
    
        return subaxes
    
    subaxs = _create_subsubplots(ax, scalefactor='46%', texts=['Upper PI','Lower PI','Significance'])

    z_lower = np.percentile(z_samples, percentiles[0], axis=1)
    z_upper = np.percentile(z_samples, percentiles[1], axis=1)
    # z_significance = np.sign(z_lower*z_upper)*np.sqrt(np.power(np.abs(z_lower*z_upper),np.sign(z_lower*z_upper)))
    # z_significance = np.sign(z_lower*z_upper)*np.sqrt(np.abs(z_lower*z_upper))

    from scipy.special import logit
    ε = 0.02
    z_significance = \
    -logit( (2 - ε)*np.min([(z_samples > 0).mean(axis=1),(z_samples < 0).mean(axis=1)], axis=0) + 0.25*ε )    
#    z_significance = \
#    1 - 2*np.min([(z_samples > 0).mean(axis=1),(z_samples < 0).mean(axis=1)], axis=0)

    vmax = max(z_upper.max(),-z_lower.min())
    vmin = -vmax

#    vmax2 = max(z_significance.max(), -z_significance.min())
#    vmin2 = -vmax2
    vmax2 = -logit(0.25*ε)
    vmin2 = -vmax2

    plot_unstructured_heatmap(z_upper , xy, ax=subaxs[0], show_colorbar=False, vmin = vmin, vmax = vmax, colormap=colormap, grid_res=grid_res, title='')
    plot_unstructured_heatmap(z_lower , xy, ax=subaxs[1], show_colorbar=False, vmin = vmin, vmax = vmax, colormap=colormap, grid_res=grid_res, title='')
    plot_unstructured_heatmap(z_significance , xy, ax=subaxs[2], show_colorbar=False, vmin = vmin2, vmax = vmax2, colormap='Reds_r', grid_res=grid_res, title='', use_contour=True)
    
    sharex = True
    sharey = True
    xmin, xmax = float('inf'), float('-inf')
    ymin, ymax = float('inf'), float('-inf')
    for i in range(3):
        xmin, xmax = min(xmin, subaxs[i].get_xlim()[0]), max(xmax, subaxs[i].get_xlim()[1])
        ymin, ymax = min(ymin, subaxs[i].get_ylim()[0]), max(ymax, subaxs[i].get_ylim()[1])
    if sharex or sharey:
        for i in range(3):
            subax = subaxs[i]
            if sharex:
                subax.set_xlim(xmin, xmax)
                if i == 0:
                    subax.set_xticklabels([])
            if sharey:
                subax.set_ylim(ymin, ymax)
                if i == 2:
                    subax.set_yticklabels([])

    return vmin, vmax





def convert_chain_draw_to_sample(xr_datarray):
    """
    Takes an xarray.DataArray as input where two indexes are 'chain' and 'draw', and returns it with 'sample'
    """
    return(xr_datarray.stack(sample=('chain','draw')))






def uncertain_lineplot(x, y, pi = 90, ax = None, color = '#5b8e7d'):
    """
    Plots a line graph with uncertainty intervals based on percentiles.

    The function creates a line plot using seaborn, where the central line represents
    the median of the data points, and the shaded area around it represents the uncertainty
    interval defined by the specified percentile.

    Parameters
    ----------
    x : numpy.ndarray
        A 1-dimensional numpy array representing the x-axis values of the plot. It should
        have a shape of (n_points,).
    y : numpy.ndarray
        A 2-dimensional numpy array representing multiple samples for y-axis values corresponding
        to each x-axis point. It should have a shape of (n_sample, n_points), where `n_sample`
        is the number of samples for each point on the x-axis.
    pi : int, optional
        The percentile to use for calculating the uncertainty interval around the median line.
        The default is 90, which means the plot will display the 5th to 95th percentile range
        as the uncertainty area.
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the lineplot. If None (default), the current axes will be used.
    color : str, optional
        The color to use for the line and uncertainty area in the plot. The default is '#5b8e7d'.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot drawn onto it.

    Raises
    ------
    ValueError
        If `x` or `y` do not match the expected dimensions or if `pi` is not between 0 and 100.

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = np.random.rand(100, 10)
    >>> uncertain_lineplot(x, y)

    """

    from pandas import DataFrame
    from seaborn import lineplot

    # Validation checks
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("x must be a 1-dimensional numpy array.")
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != x.size:
        raise ValueError("y must be a 2-dimensional numpy array with the same number of points as x in its second dimension.")
    if not (0 <= pi <= 100):
        raise ValueError("pi must be between 0 and 100, inclusive.")

    if ax is None:
        ax = plt.gca()

    return(
        lineplot(
            DataFrame(
                y,
                columns=x
            ).melt(),
            x='variable',
            y='value',
            errorbar=('pi',pi),
            ax=ax,
            c=color
        )
    )




def plot_Y_training(idata, sample_idx = 0, ax_1 = None, ax_2 = None, ax_heatmap_1 = None, ax_heatmap_2 = None, required = "predictive", conditional = False, plotting_3d_choice_candle = False, plotting_3d_choice_heatmap = False, scatter = True, scatter_color = '#bc4b51', band_color_lineplot = '#5b8e7d', rng_seed = 123):
    """
    Plots the training data along with predictive distributions for a multitarget latent factor model. This function
    allows for the visualization of the observed data against the predictive distributions, supporting both 2D and 3D
    plots based on the dimensionality of the observation locations.

    Parameters
    ----------
    idata : az.InferenceData
        An ArviZ InferenceData structure containing the observed data, constant data, and posterior distributions.
    sample_idx : int, optional
        Index of the sample to plot, by default 0.
    ax_1 : matplotlib.axes.Axes or None or bool, optional
        Axes object for plotting the first target variable. If None, a new figure is created. If False do not plot target 1.
    ax_2 : matplotlib.axes.Axes or None or bool, optional
        Axes object for plotting the second target variable. Similar to `ax_1`. If False do not plot target 2.
    required : str, optional
        Specifies the type of predictive distribution to plot ('predictive', 'predictive idiosyncratic', 'predictive estimate').
    conditional : bool, optional
        Determines whether to sample from the conditional or unconditional predictive distribution, by default False.
    plotting_3d_choice_candle : bool, optional
        If True, plots candlestick charts in 3D; otherwise, plots a band in 3D, by default False.
    plotting_3d_choice_heatmap : bool, optional
        If True plots 3D using many heatmaps.
    scatter : bool, optional
        If True, includes scatter plots of the observed data, by default True.
    scatter_color : str, optional
        Color for the scatter plot points, by default '#bc4b51'.
    band_color_lineplot : str, optional
        Color for the band or line plot, by default '#5b8e7d'.
    rng_seed : int, optional
        Seed for the random number generator to ensure reproducible results, by default 123.

    Raises
    ------
    ValueError
        If `required` is not one of the allowed values.
        If `conditional` is not a boolean.
        If `ax_1` and `ax_2` have incompatible types or values.
        If the dimensionality of observation locations is not supported.

    Notes
    -----
    This function integrates closely with the modeling framework, assuming specific structure and naming conventions
    in the `idata` object. It supports flexibility in plotting configurations and can visualize both 2D and 3D data,
    contingent on the dimensionality of the observation locations.

    See Also
    --------
    sample_conditional_predictive, sample_unconditional_predictive : Functions for sampling predictive distributions.
    uncertain_lineplot : Function used for lineplot error bars.
    plot_3d_with_computed_error_bars, plot_3d_with_computed_percentiles : Functions for plotting in 3D.
    """
    import matplotlib as mpl
    import arviz as az
    from .multitarget_latent_factor_model import renaming_convention, sample_unconditional_predictive, sample_conditional_predictive
    rng = np.random.default_rng(rng_seed)

    if required not in ['predictive', 'predictive idiosyncratic', 'predictive estimate']:
        raise ValueError("required must be one of 'predictive', 'predictive idiosyncratic', 'predictive estimate'.")

    if not isinstance(conditional, bool):
        raise ValueError("conditional must be a boolean.")

    if not isinstance(idata, az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData instance.")

    if not isinstance(sample_idx, int):
        raise ValueError("sample_idx must be an integer.")

    if not isinstance(plotting_3d_choice_candle, bool):
        raise ValueError("plotting_3d_choice_candle must be a boolean.")

    if not isinstance(plotting_3d_choice_heatmap, bool):
        raise ValueError("plotting_3d_choice_heatmap must be a boolean.")

    if plotting_3d_choice_candle and plotting_3d_choice_heatmap:
        raise ValueError("candle and heatmap cannot be both True.")

    if not isinstance(scatter, bool):
        raise ValueError("scatter must be a boolean.")

    if not ax_heatmap_1 is None and not isinstance(ax_heatmap_1, mpl.axes._axes.Axes):
        raise ValueError("ax_heatmap_1 must either be None or an Axes")
    if not ax_heatmap_2 is None and not isinstance(ax_heatmap_2, mpl.axes._axes.Axes):
        raise ValueError("ax_heatmap_2 must either be None or an Axes")


    L1 = int( idata.constant_data['L1'] )
    t1 = idata.constant_data['t1'].values
    if t1.shape == (L1,):
        dimensionality1 = 1
    elif t1.shape == (L1,2):
        dimensionality1 = 2
    else:
        raise ValueError("'t1' is not of shape (L1,) or (L1,2).")

    L2 = int( idata.constant_data['L2'] )
    t2 = idata.constant_data['t2'].values
    if t2.shape == (L2,):
        dimensionality2 = 1
    elif t2.shape == (L2,2):
        dimensionality2 = 2
    else:
        raise ValueError("'t2' is not of shape (L2,) or (L2,2).")


    plot_1 = True
    plot_2 = True

    if plotting_3d_choice_heatmap:
        if dimensionality1 == 2:
            if ax_1 is None or ax_heatmap_1 is None:
                raise ValueError("For heatmap plots automatic figure creation is not supported.")
            if ax_1 == False:
                plot_1 = False
        if dimensionality2 == 2:
            if ax_2 is None or ax_heatmap_2 is None:
                raise ValueError("For heatmap plots automatic figure creation is not supported.")
            if ax_2 == False:
                plot_2 = False

    if ax_1 is None and ax_2 is None:
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        ax_1 = axs[0]
        ax_2 = axs[1]
    elif ax_1 is None and ax_2 == False:
        fig, ax_1 = plt.subplots(1,1, figsize=(7,7))
        plot_2 = False
    elif ax_1 == False and ax_2 is None:
        fig, ax_1 = plt.subplots(1,1, figsize=(7,7))
        plot_1 = False
    elif isinstance(ax_1, mpl.axes._axes.Axes) and isinstance(ax_2, mpl.axes._axes.Axes):
        pass
    elif isinstance(ax_1, mpl.axes._axes.Axes) and ax_2 == False:
        plot_2 = False
    elif ax_1 == False and isinstance(ax_2, mpl.axes._axes.Axes):
        plot_1 = False
    else:
        raise ValueError("ax_1 and ax_2 should either be both None, both Axes, or boolean and Axes, or boolean and None")



    y1 = renaming_convention( idata.observed_data['y1'] ).sel(sample_idx = sample_idx).values
    y2 = renaming_convention( idata.observed_data['y2'] ).sel(sample_idx = sample_idx).values

    X = renaming_convention( idata.constant_data['X'] ).sel(sample_idx = sample_idx).values

    if conditional:
        if plot_1:
            y1_predictive = \
            sample_conditional_predictive(
                idata,
                idata.constant_data['X'].values[:,sample_idx][:,np.newaxis],
                rng.integers(1000000),
                bootstrap=1000,
                Y2_test = idata.observed_data['y2'].values[:,sample_idx][:,np.newaxis],
                required=required
            ).values[:,:,0].T
        if plot_2:
            y2_predictive = \
            sample_conditional_predictive(
                idata,
                idata.constant_data['X'].values[:,sample_idx][:,np.newaxis],
                rng.integers(1000000),
                bootstrap=1000,
                Y1_test = idata.observed_data['y1'].values[:,sample_idx][:,np.newaxis],
                required=required
            ).values[:,:,0].T
    else:
        xr_unconditional = \
        sample_unconditional_predictive(
            idata,
            idata.constant_data['X'].values[:,sample_idx][:,np.newaxis],
            rng.integers(1000000),
            bootstrap=1000,
            required = required
            )
        y1_predictive = xr_unconditional['Y1'].values[:,:,0].T
        y2_predictive = xr_unconditional['Y2'].values[:,:,0].T

    if plot_1:
        ax = ax_1
        if dimensionality1 == 1:
            # 2D Line plot for target 1
            uncertain_lineplot(t1, y1_predictive.T, pi=90, ax=ax, color=band_color_lineplot)
        elif dimensionality1 == 2 and not plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed percentiles for target 1
            _, ax3d = plot_3d_with_computed_percentiles(t1, y1_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality1 == 2 and plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed error bars for target 1
            _, ax3d = plot_3d_with_computed_error_bars(t1, y1_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality1 == 2 and plotting_3d_choice_heatmap:
            vmin, vmax = plot_3_subplots_uncertainty( t1, y1_predictive, ax=ax_heatmap_1)
            plot_unstructured_heatmap(y1_predictive.mean(axis=1), t1, ax=ax, colormap='coolwarm')


        # Scatter plot of observed data
        if scatter:
            if dimensionality1 == 1:
                ax.scatter(t1, y1, color=scatter_color)
            elif dimensionality1 == 2 and not plotting_3d_choice_heatmap:
                ax3d.scatter( t1[:,0], t1[:,1], y1, color=scatter_color )
            elif dimensionality1 == 2 and plotting_3d_choice_heatmap:
                ax.scatter( t1[:,0], t1[:,1], c=y1, cmap='coolwarm', edgecolors='black' )

    if plot_2:
        ax = ax_2
        if dimensionality2 == 1:
            # 2D Line plot for target 2
            uncertain_lineplot(t2, y2_predictive.T, pi=90, ax=ax, color=band_color_lineplot)
        elif dimensionality2 == 2 and not plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed percentiles for target 2
            _, ax3d = plot_3d_with_computed_percentiles(t2, y2_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality2 == 2 and plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed error bars for target 2
            _, ax3d = plot_3d_with_computed_error_bars(t2, y2_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality2 == 2 and plotting_3d_choice_heatmap:
            vmin, vmax = plot_3_subplots_uncertainty( t2, y2_predictive, ax=ax_heatmap_2)
            plot_unstructured_heatmap(y2_predictive.mean(axis=1), t2, ax=ax, colormap='coolwarm')

        # Scatter plot of observed data
        if scatter:
            if dimensionality2 == 1:
                ax.scatter(t2, y2, color=scatter_color)
            elif dimensionality2 == 2 and not plotting_3d_choice_heatmap:
                ax3d.scatter( t2[:,0], t2[:,1], y2, color=scatter_color )
            elif dimensionality2 == 2 and plotting_3d_choice_heatmap:
                ax.scatter( t2[:,0], t2[:,1], c=y2, cmap='coolwarm', edgecolors='black' )




def plot_Y_testing(idata, test_set_dic, sample_idx = 0, ax_1 = None, ax_2 = None, ax_heatmap_1 = None, ax_heatmap_2 = None, required = "predictive", conditional = False, plotting_3d_choice_candle = False, plotting_3d_choice_heatmap = False, scatter = True, scatter_color = '#bc4b51', band_color_lineplot = '#5b8e7d', rng_seed = 123):
    """
    Plots the training data along with predictive distributions for a multitarget latent factor model. This function
    allows for the visualization of the observed data against the predictive distributions, supporting both 2D and 3D
    plots based on the dimensionality of the observation locations.

    Parameters
    ----------
    idata : az.InferenceData
        An ArviZ InferenceData structure containing the observed data, constant data, and posterior distributions.
    test_set_dic: 
        A dictionary containing 'y1','y2' and 'X' for the test set.
    sample_idx : int, optional
        Index of the sample to plot, by default 0.
    ax_1 : matplotlib.axes.Axes or None or bool, optional
        Axes object for plotting the first target variable. If None, a new figure is created. If False do not plot target 1.
    ax_2 : matplotlib.axes.Axes or None or bool, optional
        Axes object for plotting the second target variable. Similar to `ax_1`. If False do not plot target 2.
    required : str, optional
        Specifies the type of predictive distribution to plot ('predictive', 'predictive idiosyncratic', 'predictive estimate').
    conditional : bool, optional
        Determines whether to sample from the conditional or unconditional predictive distribution, by default False.
    plotting_3d_choice_candle : bool, optional
        If True, plots candlestick charts in 3D; otherwise, plots a band in 3D, by default False.
    plotting_3d_choice_heatmap : bool, optional
        If True plots 3D using many heatmaps.
    scatter : bool, optional
        If True, includes scatter plots of the observed data, by default True.
    scatter_color : str, optional
        Color for the scatter plot points, by default '#bc4b51'.
    band_color_lineplot : str, optional
        Color for the band or line plot, by default '#5b8e7d'.
    rng_seed : int, optional
        Seed for the random number generator to ensure reproducible results, by default 123.

    Raises
    ------
    ValueError
        If `required` is not one of the allowed values.
        If `conditional` is not a boolean.
        If `ax_1` and `ax_2` have incompatible types or values.
        If the dimensionality of observation locations is not supported.

    Notes
    -----
    This function integrates closely with the modeling framework, assuming specific structure and naming conventions
    in the `idata` object. It supports flexibility in plotting configurations and can visualize both 2D and 3D data,
    contingent on the dimensionality of the observation locations.

    See Also
    --------
    sample_conditional_predictive, sample_unconditional_predictive : Functions for sampling predictive distributions.
    uncertain_lineplot : Function used for lineplot error bars.
    plot_3d_with_computed_error_bars, plot_3d_with_computed_percentiles : Functions for plotting in 3D.
    """

    ########################################################################
    ### TO-DO FIX THE DOCSTRING and the validation checks!!!!!
    ########################################################################

    import matplotlib as mpl
    import arviz as az
    from .multitarget_latent_factor_model import renaming_convention, sample_unconditional_predictive, sample_conditional_predictive
    rng = np.random.default_rng(rng_seed)

    if required not in ['predictive', 'predictive idiosyncratic', 'predictive estimate']:
        raise ValueError("required must be one of 'predictive', 'predictive idiosyncratic', 'predictive estimate'.")

    if not isinstance(conditional, bool):
        raise ValueError("conditional must be a boolean.")

    if not isinstance(idata, az.InferenceData):
        raise ValueError("idata must be an arviz.InferenceData instance.")

    if not isinstance(sample_idx, int):
        raise ValueError("sample_idx must be an integer.")

    if not isinstance(plotting_3d_choice_candle, bool):
        raise ValueError("plotting_3d_choice_candle must be a boolean.")

    if not isinstance(plotting_3d_choice_heatmap, bool):
        raise ValueError("plotting_3d_choice_heatmap must be a boolean.")

    if plotting_3d_choice_candle and plotting_3d_choice_heatmap:
        raise ValueError("candle and heatmap cannot be both True.")

    if not isinstance(scatter, bool):
        raise ValueError("scatter must be a boolean.")

    if not ax_heatmap_1 is None and not isinstance(ax_heatmap_1, mpl.axes._axes.Axes):
        raise ValueError("ax_heatmap_1 must either be None or an Axes")
    if not ax_heatmap_2 is None and not isinstance(ax_heatmap_2, mpl.axes._axes.Axes):
        raise ValueError("ax_heatmap_2 must either be None or an Axes")


    L1 = int( idata.constant_data['L1'] )
    t1 = idata.constant_data['t1'].values
    if t1.shape == (L1,):
        dimensionality1 = 1
    elif t1.shape == (L1,2):
        dimensionality1 = 2
    else:
        raise ValueError("'t1' is not of shape (L1,) or (L1,2).")

    L2 = int( idata.constant_data['L2'] )
    t2 = idata.constant_data['t2'].values
    if t2.shape == (L2,):
        dimensionality2 = 1
    elif t2.shape == (L2,2):
        dimensionality2 = 2
    else:
        raise ValueError("'t2' is not of shape (L2,) or (L2,2).")


    plot_1 = True
    plot_2 = True

    if plotting_3d_choice_heatmap:
        if dimensionality1 == 2:
            if ax_1 is None or ax_heatmap_1 is None:
                raise ValueError("For heatmap plots automatic figure creation is not supported.")
            if ax_1 == False:
                plot_1 = False
        if dimensionality2 == 2:
            if ax_2 is None or ax_heatmap_2 is None:
                raise ValueError("For heatmap plots automatic figure creation is not supported.")
            if ax_2 == False:
                plot_2 = False

    if ax_1 is None and ax_2 is None:
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        ax_1 = axs[0]
        ax_2 = axs[1]
    elif ax_1 is None and ax_2 == False:
        fig, ax_1 = plt.subplots(1,1, figsize=(7,7))
        plot_2 = False
    elif ax_1 == False and ax_2 is None:
        fig, ax_1 = plt.subplots(1,1, figsize=(7,7))
        plot_1 = False
    elif isinstance(ax_1, mpl.axes._axes.Axes) and isinstance(ax_2, mpl.axes._axes.Axes):
        pass
    elif isinstance(ax_1, mpl.axes._axes.Axes) and ax_2 == False:
        plot_2 = False
    elif ax_1 == False and isinstance(ax_2, mpl.axes._axes.Axes):
        plot_1 = False
    else:
        raise ValueError("ax_1 and ax_2 should either be both None, both Axes, or boolean and Axes, or boolean and None")


    y1 = test_set_dic['y1'][:,sample_idx]
    y2 = test_set_dic['y2'][:,sample_idx]
    X = test_set_dic['X'][:,sample_idx]

    if conditional:
        if plot_1:
            y1_predictive = \
            sample_conditional_predictive(
                idata,
                test_set_dic['X'][:,sample_idx][:,np.newaxis],
                rng.integers(1000000),
                bootstrap=1000,
                Y2_test = test_set_dic['y2'][:,sample_idx][:,np.newaxis],
                required=required
            ).values[:,:,0].T
        if plot_2:
            y2_predictive = \
            sample_conditional_predictive(
                idata,
                test_set_dic['X'][:,sample_idx][:,np.newaxis],
                rng.integers(1000000),
                bootstrap=1000,
                Y1_test = test_set_dic['y1'][:,sample_idx][:,np.newaxis],
                required=required
            ).values[:,:,0].T
    else:
        xr_unconditional = \
        sample_unconditional_predictive(
            idata,
            test_set_dic['X'][:,sample_idx][:,np.newaxis],
            rng.integers(1000000),
            bootstrap=1000,
            required = required
            )
        y1_predictive = xr_unconditional['Y1'].values[:,:,0].T
        y2_predictive = xr_unconditional['Y2'].values[:,:,0].T

    if plot_1:
        ax = ax_1
        if dimensionality1 == 1:
            # 2D Line plot for target 1
            uncertain_lineplot(t1, y1_predictive.T, pi=90, ax=ax, color=band_color_lineplot)
        elif dimensionality1 == 2 and not plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed percentiles for target 1
            _, ax3d = plot_3d_with_computed_percentiles(t1, y1_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality1 == 2 and plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed error bars for target 1
            _, ax3d = plot_3d_with_computed_error_bars(t1, y1_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality1 == 2 and plotting_3d_choice_heatmap:
            vmin, vmax = plot_3_subplots_uncertainty( t1, y1_predictive, ax=ax_heatmap_1)
            plot_unstructured_heatmap(y1_predictive.mean(axis=1), t1, ax=ax, colormap='coolwarm')


        # Scatter plot of observed data
        if scatter:
            if dimensionality1 == 1:
                ax.scatter(t1, y1, color=scatter_color)
            elif dimensionality1 == 2 and not plotting_3d_choice_heatmap:
                ax3d.scatter( t1[:,0], t1[:,1], y1, color=scatter_color )
            elif dimensionality1 == 2 and plotting_3d_choice_heatmap:
                ax.scatter( t1[:,0], t1[:,1], c=y1, cmap='coolwarm', edgecolors='black' )

    if plot_2:
        ax = ax_2
        if dimensionality2 == 1:
            # 2D Line plot for target 2
            uncertain_lineplot(t2, y2_predictive.T, pi=90, ax=ax, color=band_color_lineplot)
        elif dimensionality2 == 2 and not plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed percentiles for target 2
            _, ax3d = plot_3d_with_computed_percentiles(t2, y2_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality2 == 2 and plotting_3d_choice_candle and not plotting_3d_choice_heatmap:
            # 3D Plot with computed error bars for target 2
            _, ax3d = plot_3d_with_computed_error_bars(t2, y2_predictive, percentiles=(5, 95), ax=ax)
        elif dimensionality2 == 2 and plotting_3d_choice_heatmap:
            vmin, vmax = plot_3_subplots_uncertainty( t2, y2_predictive, ax=ax_heatmap_2)
            plot_unstructured_heatmap(y2_predictive.mean(axis=1), t2, ax=ax, colormap='coolwarm')

        # Scatter plot of observed data
        if scatter:
            if dimensionality2 == 1:
                ax.scatter(t2, y2, color=scatter_color)
            elif dimensionality2 == 2 and not plotting_3d_choice_heatmap:
                ax3d.scatter( t2[:,0], t2[:,1], y2, color=scatter_color )
            elif dimensionality2 == 2 and plotting_3d_choice_heatmap:
                ax.scatter( t2[:,0], t2[:,1], c=y2, cmap='coolwarm', edgecolors='black' )





def plot_with_credibility_intervals(X, Y, ax=None, show_means = False, pi=0.95):
    """
    Plots credibility intervals for multiple sets of predictions alongside their actual values. This function is designed to visualize the uncertainty in predictions by plotting the 95% credibility intervals and, optionally, the means of the predictions. 

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array of shape (L, k) representing the actual values of k variables across L instances.
    Y : numpy.ndarray
        A 3D array of shape (S, L, k) representing S sets of predictions for k variables across L instances. These predictions are used to calculate the credibility intervals.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, the current axes will be used.
    show_means : bool, optional
        If True, plot the means of the predictions alongside the actual values and credibility intervals. Defaults to False.
    pi : float, optional
        Amplitude of the Credibility Intervals. Defaults to 0.95.

    Raises
    ------
    ValueError
        If `X` is not a 2D array or if `Y` is not a 3D array.
        If the shapes of `X` and `Y` are incompatible, i.e., if `X.shape` does not match `(Y.shape[1], Y.shape[2])`.

    Notes
    -----
    The function plots the actual values from `X` and the corresponding 95% credibility intervals derived from `Y`. For each variable (column in `X` and last dimension in `Y`), the function plots a series of vertical lines representing the credibility intervals for each instance. If `show_means` is True, it also plots the means of the predictions as scatter points.
    """

    # Check if X is a 2D array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("`X` must be a 2D numpy array.")

    # Check if Y is a 3D array
    if not isinstance(Y, np.ndarray) or Y.ndim != 3:
        raise ValueError("`Y` must be a 3D numpy array.")

    # Check if the shapes of X and Y are compatible
    if X.shape[0] != Y.shape[1] or X.shape[1] != Y.shape[2]:
        raise ValueError("The shape of `X` must match `(Y.shape[1], Y.shape[2])`.")

    upp_bound_percentile_val = pi + (1-pi)/2
    upp_bound_percentile_val = upp_bound_percentile_val*100
    low_bound_percentile_val = 100 - upp_bound_percentile_val

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    
    # Dimensions based on X and Y
    L, k = X.shape
    S = Y.shape[0]

    # Flatten X and prepare x-axis for scatter plot
    X_flattened = X.flatten()
    x_indices = np.arange(len(X_flattened))

    # Scatter plot of X flattened
    rng_aux = np.random.default_rng(123)
    colors = [default_color_dict[name] for name in list(rng_aux.choice( np.array(list(default_color_dict)) , replace=False, size=len(list(default_color_dict))))]
    
    # Compute and plot credibility intervals with line segments
    n_X_is_within_range = 0
    for i in range(k):
        # Calculating the 2.5th and 97.5th percentiles for each column in Y, which are our credibility intervals
        lower_bound = np.percentile(Y[:,:,i], low_bound_percentile_val, axis=0)
        upper_bound = np.percentile(Y[:,:,i], upp_bound_percentile_val, axis=0)

        # Plotting the intervals as vertical line segments
        for j in range(L):
            if show_means:
                ax.scatter([j + L*i], [X[j,i]], color='black', label=f'X[:,{i}]')
                ax.scatter([j + L*i], [Y[:,j,i].mean()], color=colors[i%k], label=f'X[:,{i}]')
            else:
                ax.scatter([j + L*i], [X[j,i]], color=colors[i%k], label=f'X[:,{i}]')
                
            ax.plot([j + L*i, j + L*i], [lower_bound[j], upper_bound[j]], color=colors[i%k])
            if lower_bound[j] < X[j,i] and X[j,i] < upper_bound[j]:
                n_X_is_within_range += 1

    ax.axhline(0, color='black', linestyle='--')

    print('X is within range ' + str(n_X_is_within_range) + ' times out of ' + str(L*k) + '. Which means ' + str(n_X_is_within_range/(L*k)*100) + '% of the time.')
