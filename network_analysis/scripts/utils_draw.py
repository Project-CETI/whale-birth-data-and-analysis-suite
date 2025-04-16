import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
import xgi
from shapely import convex_hull
from shapely.ops import unary_union

sb.set_theme(style="ticks", context="paper")


__all__ = [
    "plot_convex_hull",
    "plot_series_distribution",
    "plot_time_series_by_window_fill_between",
    "calculate_window_widths", 
    "remove_middle_spines", 
    "add_slanted_break", 
    "plot_time_series_by_window",
    "plot_frame",
    "plot_orientation_arrows",
    "compute_fig_lims",
    "plot_presence",
    "plot_orientations",
    "plot_trajectories",
    "plot_graph",
    "plot_graphx",
    "plot_masks",
]

def plot_convex_hull(seg, frame, ax=None, **kwargs):
    """
    Plots the convex hull of polygons from a given frame onto a Matplotlib axis.

    Parameters
    ----------
    seg : dict
        Dictionary containing polygon data. Expected to have a key `"masks_polygons"`, 
        where `seg["masks_polygons"][frame]` is a list of Shapely polygons.
    frame : int
        The index of the frame for which to compute the convex hull.
    ax : matplotlib.axes.Axes, optional
        The Matplotlib axis on which to plot. If `None`, the current axis (`plt.gca()`) is used.
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.plot()`, such as color or linestyle.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object with the plotted convex hull.

    """
    if ax is None:
        ax = plt.gca()

    merged = unary_union(seg["masks_polygons"][frame])

    # Compute the convex hull
    convex_hull = merged.convex_hull
    
    hx, hy = convex_hull.exterior.xy
    hx = np.array(hx)
    hy = np.array(hy)
    
    ax.plot(hx, -hy, **kwargs)

    return ax

def plot_series_distribution(series, times, time_phase_mapping, show_tests=False, test="t-test_ind", 
    test_pairs=[("pre", "post")], ax=None, **kwargs):
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        'Time': times,
        'Value': series,
        'Phase': [time_phase_mapping(t) for t in times]
    })

    data = data.dropna() # NaNs mess up the stats

    # Create a plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    #Plot the distribution
    sb.boxplot(
        x='Phase', y='Value', data=data, ax=ax, showfliers=False, fill=False, color="k", zorder=2, **kwargs
    )
    sb.stripplot(
        x='Phase', y='Value', data=data, ax=ax, 
        hue='Phase', size=4, jitter=True, alpha=0.4, zorder=-2, rasterized=True, facecolor="w"
    )

    # sb.violinplot(
    #     x='Phase', y='Value', data=data, ax=ax, fill=False, inner=None, color="k"
    # )
    
    # Add labels

    if show_tests:
        from statannotations.Annotator import Annotator
        annotator = Annotator(
            ax,
            pairs=test_pairs,
            data=data,
            x="Phase",
            y="Value",
        )
        annotator.configure(
            test=test, text_format="simple", loc="outside", verbose=True, line_width=1, 
            show_test_name=False, comparisons_correction="bonferroni"
        )
        annotator.apply_and_annotate()
    
    return ax


def plot_time_series_by_window_fill_between(series, times, std, colors=None, axs=None, time_windows=None, time_phase_mapping=None, show_legend=True,**kwargs):
    """
    Plot time series segmented into windows with color changes based on phases.

    Parameters
    ----------
    series : np.ndarray
        Time series data to plot.
    times : np.ndarray
        Corresponding time points for the series.
    colors : list of str, optional
        List of colors for the phases. Default is None.
    axs : list of matplotlib.axes.Axes, optional
        Predefined axes to plot on. Default is None.
    time_windows : list of tuple
        List of (start, end) tuples defining time windows. Default is None.
    time_phase_mapping : function
        Function mapping a time to its phase (str). Default is None.
    show_legend : bool, optional
        Whether to display a legend for the phases. Default is True.
    **kwargs : dict
        Additional keyword arguments for the plot function.

    Returns
    -------
    tuple
        Figure and axes objects.
    """
    if time_windows is None or len(time_windows) == 0:
        raise ValueError("time_windows must be a non-empty list of (start, end) tuples.")
    if time_phase_mapping is None:
        raise ValueError("time_phase_mapping must be provided.")
    
    unique_phases = ["pre", "during", "post", "later"] #set(time_phase_mapping(t) for t in times)
    n_phases = len(unique_phases)
    
    # set colors 
    if colors is None:
        palette = sb.color_palette()
        phase_to_color = {phase: palette[i] for i, phase in enumerate(unique_phases)}
    else:
        phase_to_color = {phase: colors[i % len(colors)] for i, phase in enumerate(unique_phases)}

    phases = np.array([time_phase_mapping(t) for t in times])
    colors = np.array([phase_to_color[time_phase_mapping(t)] for t in times])

    # create figure layout and axs    
    widths = calculate_window_widths(time_windows)
    
    if axs is None:
        fig, axs = plt.subplots(
            nrows=1, ncols=len(time_windows), 
            figsize=(15, 5), gridspec_kw={'width_ratios': widths}, 
            sharey=True
        )
    else:
        fig = axs[0].get_figure()
    
    if len(time_windows) == 1:
        axs = [axs]

    # plot data    
    for i, (ax, (start, end)) in enumerate(zip(axs, time_windows)): # iterate over windows
        mask_time = (times >= start) & (times <= end)

        for j, phase in enumerate(unique_phases): # iterate over phases in window

            mask_phase = (phase == phases)

            mask = mask_time * mask_phase

            window_times = times[mask]
            window_series = series[mask]
            window_std = std[mask]
            window_colors = phase_to_color[phase] 
            
            ax.fill_between(
                window_times, 
                window_series-window_std, 
                window_series+window_std,
                color=window_colors, 
                alpha=0.3,
                **kwargs
            )
        
        # style adjustments
        ax.set_xlim(start, end)

        remove_middle_spines(ax, left=(i != 0), right=(i != len(time_windows) - 1), yticks=(i != 0))
        add_slanted_break(ax, left=(i != 0), right=(i != len(time_windows) - 1), angle=50, markersize=8)

    axs[-1].tick_params(axis='y', labelright=False)
    
    # add legend
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=4, label=phase) 
        for phase, color in phase_to_color.items()
    ]
    if show_legend:
        fig.legend(handles=legend_handles, loc='upper center', ncol=len(unique_phases), bbox_to_anchor=(0.5, 1.02))
    
    return fig, axs


def calculate_window_widths(time_windows):
    """
    Calculate proportional widths for axes based on time window lengths.

    Parameters
    ----------
    time_windows : list of tuple
        List of (start, end) tuples representing time windows.

    Returns
    -------
    list of float
        Proportional widths of each time window relative to the total length.
    """
    window_lengths = [end - start for start, end in time_windows]
    total_length = sum(window_lengths)
    return [length / total_length for length in window_lengths]

def remove_middle_spines(ax, left=True, right=True, yticks=True):
    """
    Remove vertical spines and optionally y-ticks for middle axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to modify.
    left : bool, optional
        Remove the left spine if True. Default is True.
    right : bool, optional
        Remove the right spine if True. Default is True.
    yticks : bool, optional
        Remove y-ticks if True. Default is True.
    """
    sb.despine(ax=ax, left=left, right=right, top=False)
    if yticks:
        ax.yaxis.set_ticks_position('none')

def add_slanted_break(ax, right=False, left=False, angle=45, markersize=12):
    """
    Add slanted breaks to indicate continuity between axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to modify.
    right : bool, optional
        Add a break on the right side if True. Default is False.
    left : bool, optional
        Add a break on the left side if True. Default is False.
    angle : float, optional
        Angle of the slanted line in degrees. Default is 45.
    """
    # Convert angle to radians and calculate slope
    rad = np.radians(angle)
    slope = np.tan(rad)

    # Define marker coordinates based on the slope
    marker = [(-1, -slope), (1, slope)]
    
    # Common plot settings
    kwargs = dict(
        marker=marker, markersize=markersize, linestyle="none",
        color='k', mec='k', mew=1, clip_on=False
    )
    if right:
        ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
    if left:
        ax.plot([0, 0], [0, 1], transform=ax.transAxes, **kwargs)


def plot_time_series_by_window(series, times, colors=None, axs=None, time_windows=None, time_phase_mapping=None, show_legend=True, **kwargs):
    """
    Plot time series segmented into windows with color changes based on phases.

    Parameters
    ----------
    series : np.ndarray
        Time series data to plot.
    times : np.ndarray
        Corresponding time points for the series.
    colors : list of str, optional
        List of colors for the phases. Default is None.
    axs : list of matplotlib.axes.Axes, optional
        Predefined axes to plot on. Default is None.
    time_windows : list of tuple
        List of (start, end) tuples defining time windows. Default is None.
    time_phase_mapping : function
        Function mapping a time to its phase (str). Default is None.
    show_legend : bool, optional
        Whether to display a legend for the phases. Default is True.
    **kwargs : dict
        Additional keyword arguments for the plot function.

    Returns
    -------
    tuple
        Figure and axes objects.
    """
    if time_windows is None or len(time_windows) == 0:
        raise ValueError("time_windows must be a non-empty list of (start, end) tuples.")
    if time_phase_mapping is None:
        raise ValueError("time_phase_mapping must be provided.")
    
    unique_phases = ["pre", "during", "post", "later"] #set(time_phase_mapping(t) for t in times)
    n_phases = len(unique_phases)
    
    # set colors 
    if colors is None:
        palette = sb.color_palette()
        phase_to_color = {phase: palette[i] for i, phase in enumerate(unique_phases)}
    else:
        phase_to_color = {phase: colors[i % len(colors)] for i, phase in enumerate(unique_phases)}

    phases = np.array([time_phase_mapping(t) for t in times])
    colors = np.array([phase_to_color[time_phase_mapping(t)] for t in times])

    # create figure layout and axs    
    widths = calculate_window_widths(time_windows)
    
    if axs is None:
        fig, axs = plt.subplots(
            nrows=1, ncols=len(time_windows), 
            figsize=(15, 5), gridspec_kw={'width_ratios': widths}, 
            sharey=True
        )
    else:
        fig = axs[0].get_figure()
    
    if len(time_windows) == 1:
        axs = [axs]

    # plot data    
    for i, (ax, (start, end)) in enumerate(zip(axs, time_windows)): # iterate over windows
        mask_time = (times >= start) & (times <= end)

        for j, phase in enumerate(unique_phases): # iterate over phases in window

            mask_phase = (phase == phases)

            mask = mask_time * mask_phase

            window_times = times[mask]
            window_series = series[mask]
            window_colors = phase_to_color[phase] 
            
            ax.plot(
                window_times, 
                window_series, 
                color=window_colors, 
                **kwargs
            )
        
        # style adjustments
        ax.set_xlim(start, end)

        remove_middle_spines(ax, left=(i != 0), right=(i != len(time_windows) - 1), yticks=(i != 0))
        add_slanted_break(ax, left=(i != 0), right=(i != len(time_windows) - 1), angle=50, markersize=8)

    axs[-1].tick_params(axis='y', labelright=False)
    
    # add legend
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=4, label=phase) 
        for phase, color in phase_to_color.items()
    ]
    if show_legend:
        fig.legend(handles=legend_handles, loc='upper center', ncol=len(unique_phases), bbox_to_anchor=(0.5, 1.02))
    
    return fig, axs



def plot_frame(
    seg,
    frame_id,
    vid_number=None,
    ax=None,
    show_box=True,
    show_centroid=True,
    show_arrow=True,
    show_label=True,
    auto_scale=True,
    ids=None,
):

    """
    Draw bounding boxes and centroids for a specific frame.

    Parameters
    ----------
    segmentations : Segmentation object
        Contains data from video segmentation
    frame_id : int
        The index of the frame to draw.
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw. If None, the current axes will be used.
    box : bool, optional
        Whether to draw bounding boxes. Default is True.
    arrow : bool, optional
        Whether to draw arrows representing orientation. Default is True.
    label : bool, optional
        Whether to label centroids with whale indices. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the drawn elements.
    """

    fps = 30  # number of frame per seconds
    time = frame_id / fps  # time in seconds

    if ax is None:
        ax = plt.gca()

    bounding_boxes_4xy_reshaped = seg["bounding_boxes"]
    centroids_xy = seg["centroids"]
    orientations_rad = seg["orientations_rad"]
    orientations_confidence = seg["orientations_confidence"]

    num_frames, num_whales = orientations_rad.shape

    for whale_idx in range(num_whales):

        # flip y-axis to match image coord
        bbox = bounding_boxes_4xy_reshaped[frame_id, whale_idx] * [1, -1]
        centroid = centroids_xy[frame_id, whale_idx] * [1, -1]

        angle = orientations_rad[frame_id, whale_idx]
        angle_confidence = orientations_confidence[frame_id, whale_idx]

        if not (
            np.all(centroid == 0) or np.all(np.isnan(centroid))
        ):  # check it is a valid box

            if show_box:
                rect = patches.Polygon(
                    bbox, closed=True, linewidth=1, edgecolor="k", facecolor="none"
                )
                ax.add_patch(rect)

            if show_centroid:
                ax.scatter(centroid[0], centroid[1])

            if show_label:
                label = whale_idx if ids is None else ids[whale_idx]
                ax.text(centroid[0], centroid[1] * 1.01, f"{label}", zorder=5)

            if show_arrow:
                radius = 100
                arrow_end = centroid + radius * np.array([np.cos(angle), np.sin(angle)])

                arrow = patches.FancyArrowPatch(
                    centroid,
                    arrow_end,
                    mutation_scale=10,
                    arrowstyle="->",
                    color="r",
                    # line below does not work because some confidence values > 1
                    # alpha=min(angle_confidence, 1) # more transparent if less confident
                )

                ax.add_patch(arrow)

    title = f"video *{str(vid_number)[-4:]} " if vid_number else ""
    title += f"frame {frame_id}/{num_frames}, {time:.2f} s"

    ax.set_title(title)

    if auto_scale:
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()
    ax.set_aspect("equal")

    return ax


def plot_orientation_arrows(seg, frame, ax=None, color="r", radius=100, **kwargs):

    if ax is None:
        ax=plt.gca()

    orientations_rad = seg["orientations_rad"]
    orientations_confidence = seg["orientations_confidence"]
    centroids_xy = seg["centroids"]

    num_frames, num_whales = orientations_rad.shape

    for whale_idx in range(num_whales):

        angle = orientations_rad[frame, whale_idx]
        centroid = centroids_xy[frame, whale_idx] * [1, -1]

        if not (
            np.all(angle == 0) or np.all(np.isnan(angle))
        ):  # check it is a valid box

            arrow_end = centroid + radius * np.array([np.cos(angle), np.sin(angle)])

            arrow = patches.FancyArrowPatch(
                centroid,
                arrow_end,
                mutation_scale=10,
                arrowstyle="->",
                color=color,
                **kwargs
                # line below does not work because some confidence values > 1
                # alpha=min(angle_confidence, 1) # more transparent if less confident
            )

            ax.add_patch(arrow)

    return ax


def compute_fig_lims(centroids_xy, fraction=0.2):
    """
    Compute the static figure limits for a whole video.

    This function calculates the figure limits based on the centroids of the segmentations.
    It considers a fraction of the width and height of the figure to determine the margin.

    Parameters
    ----------
    segmentations : Segmentation object
        The segmentation object containing centroids.
    fraction : float, optional
        Fraction of margin to be added to the figure limits, defaults to 0.2.

    Returns
    -------
    tuple
        A tuple containing the figure limits (xmin, ymin, xmax, ymax).

    """
    # centroids_xy = segmentations.get_all_centroids_xy()
    centroids_xy_flip = np.array(centroids_xy) * [1, -1]

    # replace zero by nan to avoid min = 0
    mask_zero = np.isclose(centroids_xy_flip, 0)
    centroids_nan = np.where(mask_zero, np.nan, centroids_xy_flip)

    xmax, ymax = np.nanmax(centroids_nan, axis=(0, 1))
    xmin, ymin = np.nanmin(centroids_nan, axis=(0, 1))

    width = xmax - xmin
    height = ymax - ymin
    xmargin = width * fraction
    ymargin = height * fraction

    return (xmin - xmargin, ymin - ymargin, xmax + xmargin, ymax + ymargin)


def plot_presence(seg, ax=None, title="Presence plot"):
    """
    Plot binary presence of bounding boxes (whales) over time.

    Parameters
    ----------
    segmentations : Segmentation
        Object containing segmentation data.
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axes used for plotting.
    """

    if ax is None:
        ax = plt.gca()

    # centroids_xy = segmentations.get_all_centroids_xy()
    # centroids_xy = np.array(centroids_xy)
    is_absent = np.all(seg['centroids'] == 0, axis=-1) | np.all(
        np.isnan(seg['centroids']), axis=-1
    )

    colors = ["silver", "navy"]
    cmap = sb.color_palette(colors, as_cmap=True)

    sb.heatmap(is_absent.T, cmap=cmap, cbar=False, ax=ax)

    legend_labels = ["Present", "Not Present"]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]

    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1, 0.5), loc="center left")

    #ax.set_ylabel("Whale ID")
    ax.set_xlabel("Time frame")
    ax.set_yticks(np.arange(len(seg['ids'])) + 0.5)
    ax.set_yticklabels(seg['ids'], rotation=0)

    ax.set_title(title)

    return ax


def plot_orientations(seg, axes=None, label=""):
    """
    Plot whale orientations over time.

    Parameters
    ----------
    seg : dict
        Dictionary containing segmentation data.
    axes : numpy.ndarray or None, optional
        Array of matplotlib axes. If None, new axes will be created.
    label : str
        Label the curves, for plotting a legend if plotting several curves 
        on a single set of axes.

    Returns
    -------
    axes : numpy.ndarray
        Array of matplotlib axes used for plotting.
    """

    # Calculate the number of rows for subplots
    # num_orientations = np.count_nonzero(orientations[:,:,0].mean(axis=0)) # this line is useless right?
    num_orientations = seg['orientations_rad'].shape[1]
    if num_orientations <= 0:
        return None

    if axes is None:
        num_cols = 5
        num_rows = math.ceil(num_orientations / num_cols)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(10, num_rows * 1.7), sharey=True, sharex=True
        )
    else:
        num_rows, num_cols = axes.shape
        fig = axes.ravel()[0].get_figure()

    for i, ax in enumerate(axes.ravel()):

        if i >= num_orientations:
            try:
                ax.remove() # won't work if already removed
            except Exception as e:
                pass #print(e)
            continue
    
        ax.plot(seg["frame_indices"], seg['orientations_rad'][:, i], label=label)
        # ax.set_ylabel("Orientation (rad)")
        # ax.set_xlabel("Time frame")
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels(["0", "π", "2π"])
        ax.set_title(seg['ids'][i])



    plt.suptitle("Orientations", fontsize="x-large")
    fig.supylabel("Orientations (rad)")
    #fig.supxlabel("Time frame")
    plt.tight_layout()

    return axes


def plot_trajectories(seg, label_start=True, s=10, alpha=0.4, ax=None, **kwargs):
    """
    Plot whale trajectories (centroid) aggregated over time.

    Parameters
    ----------
    seg : dict
        Dictionary containing segmentation data.
    label_start : bool, optional
        If True, labels the starting points of trajectories with whale IDs. Default is True.
    s : float, optional
        Marker size for scatter plot. Default is 10.
    alpha : float, optional
        Transparency of markers. Default is 0.4.
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axes to plot on. If None, the current axes will be used.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to `ax.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axes used for plotting.
    """

    if ax is None:
        ax = plt.gca()

    fps = 30

    # flip y-values to match drone image reference frame
    centroids = np.array(seg['centroids']) * [1, -1]

    num_frames, num_whales, _ = centroids.shape
    times = np.arange(num_frames) / fps

    for i in range(num_whales):

        centroid = centroids[:, i, :]

        data = {"t": times, "x": centroid[:, 0], "y": centroid[:, 1]}
        f = pd.DataFrame(data)
        # f['x'] = f['x'].rolling(60).median()
        # f['y'] = f['y'].rolling(60).median()
        f["f_id"] = i

        if i != 0:
            df = pd.concat((df, f), axis=0)
        else:
            df = f

    ax.scatter(df["x"], df["y"], c=df["f_id"].values, s=s, alpha=alpha, **kwargs)
    ax.set_aspect("equal")

    if label_start:
        # get colors used in scatter
        ccmap = ax.collections[0].get_cmap()
        color_mapped_values = ax.collections[0].get_array()
        try:
            vals = np.array(list(set(color_mapped_values))) / max(color_mapped_values)
            colors = ccmap(vals)
        except Exception:
            colors = ["black"] * num_whales

        for i in range(num_whales):
            centroid = centroids[:, i, :]
            t = ax.text(centroid[0, 0], centroid[0, 1], seg['ids'][i], color=colors[i])
            t.set_bbox(
                dict(facecolor="white", alpha=0.8, edgecolor="white", boxstyle="round")
            )

    ax.set_aspect("equal")

    return ax


def plot_graph(adj, seg, frame, ax=None, **kwargs):
    """
    Plot a graph representation a segmentation.

    Parameters
    ----------
    adj : array-like
        Adjacency matrix representing the graph.
    seg : dict
        Segmentation data containing information about centroids and ids.
    frame : int
        Frame index indicating which frame to plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the graph on. If None, the current axes will be used. Default is None.
    **kwargs
        Additional keyword arguments passed to xgi.draw function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plotted graph.
    cols : list
        Collections for colorbar plotting
    """
    
    if ax is None:
        ax = plt.gca()
    
    G = nx.from_numpy_array(np.matrix(adj[frame]), create_using=nx.Graph)

    H = xgi.Hypergraph()

    H.add_nodes_from(list(G.nodes))
    H.add_edges_from(list(G.edges))

    pos = {i: (xy[0], -xy[1]) for i, xy in enumerate(seg["centroids"][frame])}

    ax, cols = xgi.draw(
        H,
        pos=pos,
        node_labels=False,
        **kwargs
    )
    
    xgi.draw_node_labels(
        H,
        pos,
        node_labels={i: el for i, el in enumerate(seg["ids"])},
        #font_color_nodes="red",
        font_weight_nodes="bold",
        verticalalignment_nodes="bottom",
        horizontalalignment_nodes="right",
        clip_on_nodes=False
    )

    return ax, cols

def plot_graphx(adj, seg, frame, center_type="centroids", node_size=9, node_color="k", edge_color="grey", width=5, ax=None, **kwargs):


    if ax is None:
        ax = plt.gca()


    G = nx.Graph(adj)

    if center_type=="centroids":
        centers = seg["centroids"][frame]
    elif center_type=="masks":

        masks_poly = seg["masks_polygons"][frame]
        centers = [np.average(np.array(maskk.exterior.xy), axis=1) for maskk in masks_poly]
        centers = np.array(centers)
    else: 
        raise ValueError("center_type needs to be either 'centroids' or 'masks'.")

    centers = centers #* np.array([1, -1])

    # plot network
    pos = {i: (xy[0], -xy[1]) for i, xy in enumerate(centers)}
    nodes = nx.draw_networkx_nodes(G, pos=pos, label=False, node_size=node_size, node_color=node_color, ax=ax)
    edges = nx.draw_networkx_edges(G, pos=pos, label=False, width=width, edge_color=edge_color, ax=ax)

    nodes.set_zorder(2)
    edges.set_zorder(1)
    
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    return ax


def plot_masks(seg, frame, show_labels=True, colors=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")

    num_whales = seg["num_whales"]

    masks_poly_t = seg["masks_polygons"]

    if colors is None:
        colors = [None] * num_whales
    elif isinstance(colors, str):
        colors = [colors] * num_whales
    elif len(colors)>1:
        if len(colors)!=num_whales:
            raise ValueError("The number of colors must be equal to the number of whales")

    for i in range(num_whales):
        
        maskk = masks_poly_t[frame][i]

        xs = np.array(maskk.exterior.xy[0])
        ys = np.array(maskk.exterior.xy[1]) * (-1)

        ax.plot(xs, ys, color=colors[i], **kwargs)

                
        if show_labels:
            label = seg['ids'][i]
            center = np.average(np.array(maskk.exterior.xy), axis=1) * [1, -1]

            if not np.all(np.isnan(center)):
                ax.text(center[0],center[1], f"{label}")

    return ax
