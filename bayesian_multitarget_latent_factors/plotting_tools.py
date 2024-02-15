import numpy as np
import matplotlib.pyplot as plt


__all__ = ['plot_unstructured_heatmap',]


def plot_unstructured_heatmap(values, locations, method='cubic', grid_res=100, plot_points=False,
                              ax=None, xlabel=None, ylabel=None, title='Heatmap of Interpolated Values',
                              colorbar_label='Interpolated Value', show_colorbar=True,
                              colormap='viridis', max_distance=None, p_NN_distance_max_distance = _np.inf, **kwargs):
    """
    Plots a heatmap from unstructured data points using interpolation on an optional specified axes,
    with options for customizing the appearance. Leaves white spaces in areas without nearby evaluated points,
    controlled by the max_distance parameter.

    Parameters:
    - values: numpy array of shape (n,), the values at each location.
    - locations: numpy array of shape (n, 2), the x and y coordinates for each value.
    - method: str, the interpolation method ('linear', 'nearest', 'cubic').
    - grid_res: int or complex, the resolution of the grid for interpolation.
    - plot_points: bool, if True, the original data points are plotted over the heatmap.
    - ax: matplotlib.axes.Axes, the axes on which to plot the heatmap. If None, uses the current axis.
    - xlabel, ylabel, title: str, labels for the axes and the title of the plot.
    - colorbar_label: str, the label for the colorbar.
    - show_colorbar: bool, if True, displays the colorbar.
    - colormap: str, the colormap palette name.
    - max_distance: float, maximum distance to nearest evaluated point to perform interpolation. None disables this feature.
    - p_NN_distance_max_distance: the p of the Minkowsy distance of choice
    - **kwargs: additional keyword arguments passed to `imshow`.

    Returns:
    - ax: The matplotlib Axes object for the plot.
    - img: The QuadMesh object from imshow, useful for further customization.
    - cbar: The Colorbar object, if created. None otherwise.
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

    # Create plot on the specified axes
    if ax is None:
        ax = plt.gca()

    img = ax.imshow(grid_z.T, extent=(min(locations[:, 0]), max(locations[:, 0]), min(locations[:, 1]), max(locations[:, 1])),
                    origin='lower', aspect='auto', cmap=colormap, **kwargs)
    cbar = None
    if show_colorbar:
        cbar = plt.colorbar(img, ax=ax, label=colorbar_label)
    if plot_points:
        ax.plot(locations[:, 0], locations[:, 1], 'k.', markersize=5, alpha=0.5)  # plot the original points
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax, img, cbar




    