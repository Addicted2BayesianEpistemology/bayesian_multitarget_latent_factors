import numpy as np
import matplotlib.pyplot as plt


__all__ = ['plot_unstructured_heatmap',]


def plot_unstructured_heatmap(values, locations, method='cubic', grid_res=100, plot_points=False,
                              ax=None, xlabel=None, ylabel=None, title='Heatmap of Interpolated Values',
                              colorbar_label='Interpolated Value', show_colorbar=True,
                              vmin = None, vmax = None,
                              colormap='viridis', max_distance=None, p_NN_distance_max_distance = np.inf, **kwargs):
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

    img = ax.imshow(grid_z.T, extent=(min(locations[:, 0]), max(locations[:, 0]), min(locations[:, 1]), max(locations[:, 1])),
                    origin='lower', aspect='auto', cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)
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
        None

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
    None

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
    plot_unstructured_heatmap(z_significance , xy, ax=subaxs[2], show_colorbar=False, vmin = vmin2, vmax = vmax2, colormap='Reds_r', grid_res=grid_res, title='')
    
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




