import more_itertools as mit
import numpy as np
import shapely as spl
from PyIF import te_compute as te
from shapely import LineString

from utils_networks import num_whales_over_time

__all__ = [
    "order_parameter",
    "local_order_parameter",
    "calculate_transfer_entropy_matrix",
    "calculate_leader_follower_matrix",
    "diving_order",
    "find_stable_periods",
    "compute_view_lines",
    "compute_view_intersections",
]

def order_parameter(thetas, order=1, complex=False, axis=0):
    """
    Calculate the generalised Daido order parameter of order `order`.

    The order parameter is a measure of how the phase angles are aligned.
    It can be used to quantify the level of synchronization or coherence in a system.

    Parameters:
    -----------
    thetas : array-like, shape (n_oscillators, n_times)
        Phases of the oscillators over time
    order : int, optional
        The order of the order parameter. Default is 1.
    complex : bool, optional
        If True, return the complex order parameter. If False (default),
        return its magnitude.
    axis : int, optional
        The axis of length n_oscillators, along which the sum of the phases
        is taken. Default is 0.

    Returns:
    --------
    result : array-like
        If `complex` is True, the complex order parameter.
        If `complex` is False, the magnitude of the order parameter.
    """

    N = thetas.shape[axis]
    N_over_time = num_whales_over_time(thetas)

    Z = np.nansum(np.exp(1j * order * thetas), axis=axis) / N_over_time

    if complex:
        return Z
    else:
        return np.abs(Z)


def local_order_parameter(thetas, adjacency_matrix, order=1, axis=0, complex=False):
    """
    Calculate the local order parameter for each node in a network.

    Parameters:
    -----------
    thetas : array-like, shape (n_oscillators, n_times)
        Phases of the oscillators over time.
    adjacency_matrix : array-like, shape (n_times, n_oscillators, n_oscillators)
        Adjacency matrix representing the connections between nodes over time.
    order : int, optional
        The order of the order parameter. Default is 1.
    axis : int, optional
        The axis of length n_oscillators in thetas, along which the sum of the phases
        is taken. Default is 0.
    complex : bool, optional
        If True, return the complex order parameter. If False (default),
        return its magnitude.

    Returns:
    --------
    local_order : array-like, shape (n_oscillators,)
        Local order parameter for each node.
    """

    n_oscillators, n_times = thetas.shape if axis == 0 else thetas.shape[::-1]

    local_order = np.zeros((n_oscillators, n_times))

    for i in range(n_oscillators):
        for t in range(n_times):

            neighbors = np.nonzero(adjacency_matrix[t, i])[
                0
            ]  # Indices of neighbors of node i
            neighbor_phases = (
                thetas[neighbors, t] if axis == 0 else thetas[t, neighbors]
            )
            local_order[i, t] = np.nansum(np.exp(1j * order * neighbor_phases)) / len(
                neighbors
            )

    if complex:
        return local_order
    else:
        return np.abs(local_order)


def calculate_transfer_entropy_matrix(whale_orientations, embedding=1):
    """
    Calculate the transfer entropy matrix between whales based on their orientations.

    Parameters
    ----------
    whale_orientations : numpy.ndarray
        Array containing orientations of whales.
        It should have shape (n_times, n_whales).

    embedding : int, optional
        Embedding dimension for the transfer entropy calculation. Default is 1.

    Returns
    -------
    numpy.ndarray
        Transfer entropy matrix between whales.
        The matrix has shape (n_whales, n_whales), where element [i, j] represents
        the transfer entropy from whale j to whale i.
    """
    n_whales = whale_orientations.shape[1]
    transfer_entropy_matrix = np.zeros((n_whales, n_whales))

    for i in range(n_whales):
        for j in range(n_whales):
            if i != j:
                # Calculate transfer entropy from whale j to whale i
                transfer_entropy = te.te_compute(
                    whale_orientations[:, i],
                    whale_orientations[:, j],
                    k=1,
                    embedding=embedding,
                    safetyCheck=False,
                )
                transfer_entropy_matrix[i, j] = transfer_entropy

    return transfer_entropy_matrix


def calculate_leader_follower_matrix(whale_orientations, threshold):
    """
    Detect leader-follower relationships among whales based on transfer entropy.

    Parameters
    ----------
    whale_orientations : numpy.ndarray
        Array containing orientations of whales.
        It should have shape (n_times, n_whales).

    threshold : float
        Threshold value for detecting leader-follower relationships.
        If the transfer entropy from a whale to another whale is above this threshold,
        the relationship is considered as leader-follower.

    Returns
    -------
    numpy.ndarray
        Directionality matrix indicating leader-follower relationships among whales.
        The matrix has shape (n_whales, n_whales), where element [i, j] equals 1
        if whale i leads whale j, and 0 otherwise.
    """

    transfer_entropy_matrix = calculate_transfer_entropy_matrix(whale_orientations)
    n_whales = whale_orientations.shape[1]
    directionality_matrix = np.zeros((n_whales, n_whales))

    for i in range(n_whales):
        for j in range(n_whales):
            if i != j:
                # Check if transfer entropy is above threshold
                if transfer_entropy_matrix[i, j] > threshold:
                    directionality_matrix[i, j] = 1  # Whale i leads Whale j

    return directionality_matrix

def diving_order(seg):
    centroids = seg["centroids"][:,:,0].T
    diving_periods_all = []
    for whale in centroids:
        
        #list all frames that are nan
        diving_frames = np.where(np.isnan(whale))[0]
        
        #separate the period of diving
        diving_periods = [list(period) for period in mit.consecutive_groups(diving_frames)]
        
        #keep only the diving period >5 sec and remove the one starting under water
        diving_periods = [x for x in diving_periods if ((len(x) > 90) & (x[0]!=0))]
        diving_periods_all.append(diving_periods)
                
        
    n_whale_diving = np.sum([len(whales_n_dives) for whales_n_dives in diving_periods_all])
    whales_immersions = [(whale_t[0][0], seg['ids'][n]) for n,whale_t in enumerate(diving_periods_all) if len(whale_t)>0]
    whales_immersions = sorted(whales_immersions)
    print(f'Whales dive {n_whale_diving} out of a total of {len(centroids)} whales')
    print('The immersion order is the following:')
    for whale in whales_immersions:
        print(f'  - {whale[1]} is diving after {np.round(whale[0]/30,0)} seconds')
    print('\n')
    return whales_immersions


def find_stable_periods(seg, frame_min=0, frame_max=np.inf):
    """
    Find the periods of time during which the whales that are present don't change.

    Parameters
    ----------
    seg : dict
        Segmentation
    frame_min : int, optional
        Disregard frames lower than that values
    frame_max : int, optional
        Disregard frames higher than that values

    Returns
    -------
    stable_periods : list
        List of tuples containing the start and end indices of stable periods.
    """
    whale_presence = np.all(seg['centroids'] == 0, axis=-1) | np.all(
        np.isnan(seg['centroids']), axis=-1
    )

    stable_periods = []
    num_times, num_whales = whale_presence.shape

    # Iterate over time axis
    for t in range(num_times):
        # Check if whale presence changes from previous time step
        if t == 0 or not np.array_equal(whale_presence[t], whale_presence[t - 1]):
            start_index = t
            # Find end of stable period
            for end_index in range(t + 1, num_times):
                if not np.array_equal(whale_presence[end_index], whale_presence[t]):
                    break
            else:
                end_index = num_times
                
            if (start_index < frame_min) or (end_index > frame_max):
                continue
            stable_periods.append((start_index, end_index))

    return stable_periods



def compute_view_lines(seg, frame, r):
    """
    Compute view lines of length `r` for each whale in a given `frame`.

    For each whale, the view line start at the centroid and points in the
    direction of the orientation, with length r.

    Parameters
    ----------
    seg : dict
        Segmentation dictionary
    r : float
        The length of the view line.

    Returns
    -------
    lines : list of LineString
        A list containing LineString objects representing the view lines for each whale.
    """

    lines = []

    num_whales = seg["orientations_rad"].shape[1]

    for i in range(num_whales):
        start = seg["centroids"][frame,i] #* [1, -1]
        angle = seg["orientations_rad"][frame, i] * (-1)
        end = start + [r * np.cos(angle), r * np.sin(angle)]
        
        line = LineString([start, end])
        lines.append(line)

    return lines 


def compute_view_intersections(lines):
    """
    Compute intersection points between view lines of whales.

    Parameters
    ----------
    lines : list of LineString
        A list of LineString objects representing the view lines for each whale.

    Returns
    -------
    points : list of Point
        A list of Point objects representing the intersection points between the view lines.
    """
    points = []

    num_whales = len(lines)

    for i in range(num_whales):
        for j in range(num_whales):
            if j < i:
                if lines[i].intersects(lines[j]):
                    point = lines[i].intersection(lines[j])
                    points.append(point)
    
    return points


