import numpy as np
import shapely as spl
#import udiph
from scipy.stats import pearsonr
from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

__all__ = [
    "whales_frame_count",
    "num_whales_over_time",
    "is_invalid",
    "build_adjacency",
    "binarize_array",
    "masks_polygon_overlap",
    "masks_distances",
    "box_overlap_sym",
    "box_overlap_binary",
    "box_overlap_directed",
    "box_overlap_sym_norm",
    "distance_centroid",
    "distance_centroid_nexp",
    "binarize_matrices_percentile",
    "focus_angle",
    "zscore_matrices",
    "stochastic_renorm_matrices",
    "udiph_distance_matrix",
    "matrix_correlation",
    "construct_nearest_neighbor_networks",
]



def whales_frame_count(seg):

    return np.sum(~np.isnan(seg["orientations_rad"]), axis=0)


def num_whales_over_time(orientations_rad):
    """
    Calculate the number of whales over time based on their orientations.

    Parameters
    ----------
    orientations_rad : array-like, shape (n_frames, n_whales)
        Array containing the orientations of whales in radians over multiple frames.
        NaN values indicate a given whale is not present in a given frame.

    Returns
    -------
    counts : ndarray
        An array of shape (n_frames,) containing the count of whales with non-NaN orientations
        for each frame.
    """
    counts = np.count_nonzero(~np.isnan(orientations_rad), axis=1)
    
    return counts


def is_invalid(segmentation_element):
    """
    Check if a segmentation element (e.g. bbox, orientation, centroid) is invalid.

    A segmentation element is considered invalid if all its values are zero or NaNs.

    Parameters
    ----------
    segmentation_element : array-like
        An array representing a segmentation element.

    Returns
    -------
    bool
        True if the segmentation element is invalid, False otherwise.
    """
    return np.all(segmentation_element == 0) or np.all(np.isnan(segmentation_element))


def build_adjacency(seg, func, **kwargs):
    """
    Build a temporal adjacency matrix based on segmentations and a given similarity function.

    Parameters
    ----------
    segmentations : Segmentation
        An object containing segmentation data.
    func : function
        A similarity function between two whales.
        It can be `distance_centroid` or any of the box overlap functions.

    Returns
    -------
    ndarray
        An adjacency matrix representing the network, with shape (n_times, n_whales, n_whales).

    Notes
    -----
    When using distance centroid, values are near zero for nearby whales and larger otherwise.
    On the contrary, when using overlap, values large for nearby whales and larger otherwise.
    This should be taken into account when building a network.

    """

    bounding_boxes_4xy_reshaped = seg["bounding_boxes"]
    centroids_xy = seg["centroids"]
    orientations_rad = seg["orientations_rad"]
    orientations_confidence = seg["orientations_confidence"]
    if "masks_polygons" in seg.keys():
        masks = seg["masks_polygons"]

    num_frames, num_whales = seg["num_frames"], seg["num_whales"]

    adj = np.zeros((num_frames, num_whales, num_whales))

    for t in tqdm(range(num_frames)):

        for i in range(num_whales):

            centroid_i = centroids_xy[t, i]
            bbox_i = bounding_boxes_4xy_reshaped[t, i]
            orientation_i = orientations_rad[t, i]
            if "masks_polygons" in seg.keys():
                mask_i = masks[t][i]

            for j in range(num_whales):

                centroid_j = centroids_xy[t, j]
                bbox_j = bounding_boxes_4xy_reshaped[t, j]
                if "masks_polygons" in seg.keys():
                    mask_j = masks[t][j]

                if (func is distance_centroid) or (func is distance_centroid_nexp):
                    a_ij = func(centroid_i, centroid_j, **kwargs)
                elif (
                    (func is box_overlap_sym)
                    or (func is box_overlap_binary)
                    or (func is box_overlap_directed)
                    or (func is box_overlap_sym_norm)
                ):
                    try: 
                        a_ij = func(bbox_i, bbox_j, **kwargs)
                    except Exception as e:
                        print(e)
                        a_ij = 0
                elif (func is masks_polygon_overlap) or (func is masks_distances):
                    a_ij = func(mask_i, mask_j, **kwargs)
                elif func is focus_angle:
                    a_ij = focus_angle(centroid_i, centroid_j, orientation_i, **kwargs)
                else: 
                    raise ValueError("Something went wrong")

                adj[t, i, j] = a_ij

    return adj


def binarize_array(arr, threshold, reverse=False):
    """
    Binarize an array based on a threshold.

    Values greater than or equal to the threshold are set to 1,
    and values less than the threshold are set to 0.
    If reverse is True, the opposite is done.

    Parameters
    ----------
    arr : array_like
        The input array to be binarized.
    threshold : float
        The threshold value for binarization.
    reverse : bool, optional
        If reverse is True, values are set to 1 if less or equal
        than the threshold. Default: False.

    Returns
    -------
    ndarray
        The binarized array.

    """
    if reverse:
        return (arr <= threshold).astype(int)

    return (arr >= threshold).astype(int)


def masks_polygon_overlap(poly1, poly2, dist_treshold=15, default_value=np.nan):
    """
    Compute whether two masks are touching or not.

    Parameters
    ----------
    poly1 : shapely Polygon
        Polygon object representing a segmentation mask
    poly2 : shapely Polygon
        Polygon object representing a segmentation mask
    dist_treshold : int
        Distance in pixes under which to consider that masks 
        are touching.

    Returns
    -------
    bool
        True if overlapping
    """
    if poly1.is_empty or poly2.is_empty:
        return default_value

    return spl.distance(poly1, poly2) < dist_treshold


def masks_distances(poly1, poly2, default_value=np.nan):
    """
    Compute the distance between two masks as the min distance
    between any two points of the masks.

    Parameters
    ----------
    poly1 : shapely Polygon
        Polygon object representing a segmentation mask
    poly2 : shapely Polygon
        Polygon object representing a segmentation mask

    Returns
    -------
    float
        Smallest distance between two masks
    """
    if poly1.is_empty or poly2.is_empty:
        return default_value

    return spl.distance(poly1, poly2)



def box_overlap_sym(box1, box2, default_value=0):
    """
    Compute the area of overlap between two rectangular boxes.

    Parameters
    ----------
    box1 : array_like
        The coordinates of the first box in the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    box2 : array_like
        The coordinates of the second box in the same format as `box1`.
    default_value : float
        Value to return when at least one box is invalid (0 area). Default: 0.

    Returns
    -------
    float
        The area of overlap between the two boxes.
    """
    if is_invalid(box1) or is_invalid(box2):
        return default_value

    p1 = Polygon(box1)
    p2 = Polygon(box2)

    return p1.intersection(p2).area


def box_overlap_binary(box1, box2, default_value=False):
    """
    Whether two rectangular boxes overlap.

    Parameters
    ----------
    box1 : array_like
        The coordinates of the first box in the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    box2 : array_like
        The coordinates of the second box in the same format as `box1`.
    default_value : bool
        Value to return when at least one box is invalid (0 area). Default: False.

    Returns
    -------
    bool
        Whether two rectangular boxes overlap.
    """
    if is_invalid(box1) or is_invalid(box2):
        return default_value

    p1 = Polygon(box1)
    p2 = Polygon(box2)

    return p1.intersection(p2).area > 0


def box_overlap_directed(box1, box2, default_value=0):
    """
    Compute the area overlap between two rectangular boxes relative to the area of box1.

    Parameters
    ----------
    box1 : array_like
        The coordinates of the first box in the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    box2 : array_like
        The coordinates of the second box in the same format as `box1`.
    default_value : float
        Value to return when at least one box is invalid (0 area). Default: 0.

    Returns
    -------
    float
        The area of overlap between the two boxes relative to the area of box1.
    """
    if is_invalid(box1) or is_invalid(box2):
        return default_value

    p1 = Polygon(box1)
    p2 = Polygon(box2)

    return p1.intersection(p2).area / p1.area


def box_overlap_sym_norm(box1, box2, default_value=0):

    """
    Compute the normed area of overlap between two rectangular boxes.

    Parameters
    ----------
    box1 : array_like
        The coordinates of the first box in the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    box2 : array_like
        The coordinates of the second box in the same format as `box1`.
    default_value : float
        Value to return when at least one box is invalid (0 area). Default: 0.

    Returns
    -------
    float
        The normed area (in [0, 1]) of overlap between the two boxes.
    """
    if is_invalid(box1) or is_invalid(box2):
        return default_value

    p1 = Polygon(box1)
    p2 = Polygon(box2)

    return p1.intersection(p2).area / (p1.area + p2.area)


def distance_centroid(centroid1, centroid2, default_value=np.inf) -> float:
    """
    Compute the Euclidean distance between two centroids.

    Parameters
    ----------
    centroid1 : array_like
        The coordinates of the first centroid in the form [x1, y1].
    centroid2 : array_like
        The coordinates of the second centroid in the same format as `centroid1`.
    default_value : float
        Value to return when at least one centroid is invalid ([0,0]). Default: np.inf.

    Returns
    -------
    float
        The Euclidean distance between the two centroids.

    Note
    ----
    If used as adjacency, note that distant whales have large values (opposite to
    overlap).
    """

    if is_invalid(centroid1) or is_invalid(centroid2):
        return default_value

    return np.linalg.norm(centroid1 - centroid2)


def distance_centroid_nexp(centroid1, centroid2, default_value=0) -> float:
    """
    Compute the Euclidean distance between two centroids.
    Returns the exp(-distance) so that close whales have a value
    closer to 1. Distant whales have a value closer to 0.

    Parameters
    ----------
    centroid1 : array_like
        The coordinates of the first centroid in the form [x1, y1].
    centroid2 : array_like
        The coordinates of the second centroid in the same format as `centroid1`.
    default_value : float
        Value to return when at least one centroid is invalid ([0,0]). Default: 0.

    Returns
    -------
    float
        exp(-distance) of the distance between the two centroids.
    """

    if is_invalid(centroid1) or is_invalid(centroid2):
        return default_value

    return np.exp(-np.linalg.norm(centroid1 - centroid2))


def binarize_matrices_percentile(adj_matrices, percentile, reverse=False):
    """
    Binarize adjacency matrices based on percentiles, at each time.

    Parameters
    ----------
    adj_matrices : numpy.ndarray
        Array containing adjacency matrices.
        It should have shape (n_times, n_whales, n_whales).

    percentile : float
        Percentile value used as the threshold for binarization.

    reverse : bool, optional
        If reverse is True, values are set to 1 if less or equal
        than the threshold. Default: False.

    Returns
    -------
    numpy.ndarray
        Binarized adjacency matrices based on the percentile threshold.
        The shape remains the same as the input array.
    """

    n_times, n_whales, _ = adj_matrices.shape
    adj_matrices_bin = np.zeros_like(adj_matrices)

    for i in range(n_times):
        adj = adj_matrices[i, :, :]
        threshold = np.percentile(adj, percentile)
        adj_matrices_bin[i, :, :] = binarize_array(adj, threshold, reverse=reverse)
    return adj_matrices_bin


def focus_angle(centroid_source, centroid_target, orientation_source, default_value=np.nan):

    if is_invalid(centroid_source) or is_invalid(centroid_target):
        return default_value

    vec_to_target = centroid_target - centroid_source
    vec_to_target *= [1, -1]
    x, y = vec_to_target

    angle_to_target = np.arctan2(y, x)
    angle_to_target = np.mod(angle_to_target, 2*np.pi)

    diff_angle = angle_to_target - orientation_source
    diff_angle = np.mod(diff_angle, 2*np.pi)

    return diff_angle


def zscore_matrices(adj_matrices):
    """
    Normalize adjacency matrices by z-scoring each instant in time.

    Parameters
    ----------
    adj_matrices : numpy.ndarray
        Array containing adjacency matrices.
        It should have shape (n_times, n_whales, n_whales).

    Returns
    -------
    numpy.ndarray
        Z-scored adjacency matrices.
        The shape remains the same as the input array.
    """

    n_times, n_whales, _ = adj_matrices.shape
    adj_matrices_zscored = np.zeros_like(adj_matrices)

    for i in range(n_times):
        adj = adj_matrices[i, :, :]
        inds = np.triu_indices_from(adj, k=1)
        mu, std = np.mean(adj[inds]), np.std(adj[inds])
        adj_matrices_zscored[i, :, :] = (adj - mu) / std
    return adj_matrices_zscored


def stochastic_renorm_matrices(adj_matrices):
    """
    Normalize adjacency matrices by making them stochastic.

    Parameters
    ----------
    adj_matrices : numpy.ndarray
        Array containing adjacency matrices.
        It should have shape (n_times, n_whales, n_whales).

    Returns
    -------
    numpy.ndarray
        Stochastic adjacency matrices.
        The shape remains the same as the input array.
    """
    n_times, n_whales, _ = adj_matrices.shape
    adj_matrices_stoc = np.zeros_like(adj_matrices)

    for i in range(n_times):
        adj = adj_matrices[i, :, :]
        adj_matrices_stoc[i, :, :] = adj / np.sum(adj, axis=0)
    return adj_matrices_stoc


def udiph_distance_matrix(adj_matrices, n_neighbors=5):
    """
    Calculate UDIPH distance matrices for each time step.

    Parameters
    ----------
    adj_matrices : numpy.ndarray
        Array containing adjacency matrices.
        It should have shape (n_times, n_whales, n_whales).

    n_neighbors : int, optional
        Number of nearest neighbors for UDIPH calculation. Default is 5.

    Returns
    -------
    numpy.ndarray
        UDIPH distance matrices for each time step.
        The shape of the returned array is (n_times, n_whales, n_whales).
    """

    n_times, n_whales, _ = adj_matrices.shape
    adj_matrices_udiph = np.zeros_like(adj_matrices)

    for i in range(n_times):
        adj_matrices_udiph[i, :, :] = udiph.UDIPH(
            X=mats[i, :, :],
            n_neighbors=n_neighbors,
            distance_matrix=True,
            return_complex=False,
        )

    return adj_matrices_udiph


def matrix_correlation(matrix1, matrix2):
    """
    Calculate Pearson correlation coefficient between corresponding elements of two matrices.

    This functions ignores non-finite values (NaN or infinity) in the calculation.

    Parameters
    ----------
    matrix1 : numpy.ndarray
        First matrix for correlation calculation.

    matrix2 : numpy.ndarray
        Second matrix for correlation calculation.

    Returns
    -------
    float
        Pearson correlation coefficient between corresponding elements of matrix1 and matrix2.
    """

    # the following assumes symmetric matrices, right?
    ind1 = np.triu_indices_from(matrix1, k=1)

    fin = np.logical_and(np.isfinite(matrix1[ind1]), np.isfinite(matrix2[ind1]))

    return pearsonr(matrix1[ind1][fin], matrix2[ind1][fin])[0]


def construct_nearest_neighbor_networks(adj_matrices, n_neighbors=3):
    """
    Construct nearest neighbor networks from adjacency matrices.

    Parameters
    ----------
    adj_matrices : numpy.ndarray
        Array containing adjacency matrices.
        It should have shape (n_times, n_whales, n_whales).

    n_neighbors : int, optional
        Number of nearest neighbors to consider for constructing the network.
        Default is 3.

    Returns
    -------
    numpy.ndarray
        Nearest neighbor networks constructed from adjacency matrices.
        The shape remains the same as the input array.
    """

    n_times, n_whales, _ = adj_matrices.shape
    adj_matrices_nn = np.zeros_like(adj_matrices)

    for i in range(n_times):
        X = adj_matrices[i, :, :]
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X)

        D, idx = neigh.kneighbors()

        A = np.zeros((n_whales, n_whales))
        for i in range(n_whales):
            for j in range(n_neighbors):
                A[i, idx[i, j]] = 1

        adj_matrices_nn[i, :, :] = A

    return adj_matrices_nn


def TGD(mats, axis=0):
    tgd = np.zeros((mats.shape[axis], mats.shape[axis]))
    for i in range(mats.shape[axis]):
        for j in range(mats.shape[axis]):
            if i != j:
                tgd[i, j] = mat_corr(mats[i], mats[j])
                tgd[j, i] = tgd[i, j]
            else:
                tgd[i, j] = 0
    return tgd

