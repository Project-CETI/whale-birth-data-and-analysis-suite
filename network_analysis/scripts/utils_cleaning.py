import math
import os
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from shapely import Polygon

from utils_networks import num_whales_over_time

__all__ = [
    "smooth_orientations",
    "smooth_centroids",
    "smooth_seg_dict",
    "subtract_center_of_mass",
    "subtract_center_of_mass_seg_dict",
    "filter_seg_dict",
    "remove_nonsegmented_frames",
    "slice_seg", 
    "add_masks_polygons"
]


def smooth_orientations(orientations, window):
    """
    Smooths the orientations using a rolling median filter.

    Parameters
    ----------
    orientations : numpy.ndarray
        The input array containing orientations.
        It should have shape (n_times, n_whales).

    window : int
        The size of the sliding window for the rolling median filter.

    Returns
    -------
    numpy.ndarray
        Smoothed orientations with the same shape as the input array.
    """
    n_times, n_whales = orientations.shape

    for i in range(n_whales):
        orientations[:, i] = (
            pd.Series(orientations[:, i]).rolling(window, min_periods=1).median()
        )
    return orientations


def smooth_centroids(centroids, window):
    """
    Smooths the centroids using a rolling median filter.

    Parameters
    ----------
    centroids : numpy.ndarray
        The input array containing centroids.
        It should have shape (n_times, n_whales, 2).

    window : int
        The size of the sliding window for the rolling median filter.

    Returns
    -------
    numpy.ndarray
        Smoothed centroids with the same shape as the input array.
    """
    n_times, n_whales, _ = centroids.shape
    for i in range(n_whales):
        centroids[:, i, 0] = (
            pd.Series(centroids[:, i, 0]).rolling(window, min_periods=1).median()
        )
        centroids[:, i, 1] = (
            pd.Series(centroids[:, i, 1]).rolling(window, min_periods=1).median()
        )
    return centroids


def smooth_seg_dict(seg, window=90):
    """
    Smooth the centroids and orientations in a segmentation dictionary.

    Parameters
    ----------
    seg : dict
        The segmentation dictionary containing centroids and orientations.

    window : int, optional
        The size of the sliding window for smoothing. Default is 90.

    Returns
    -------
    dict
        A deepcopy of the input segmentation dictionary with centroids and orientations smoothed.
    """
    smoothed_seg = deepcopy(seg)
    smoothed_seg["centroids"] = smooth_centroids(smoothed_seg["centroids"], window)
    smoothed_seg["orientations_rad"] = smooth_orientations(
        smoothed_seg["orientations_rad"][:, :], window
    )

    smoothed_seg["num_whales_over_time"] = num_whales_over_time(smoothed_seg["orientations_rad"])

    return smoothed_seg


def subtract_center_of_mass(segmentation_element):
    """
    Subtract the center of mass from each segmentation element (centroid or box).

    Parameters
    ----------
    centroids : numpy.ndarray
        The input array containing centroids (n_times, n_whales, 2)
        or boxes (n_times, n_whales, 4, 2).

    Returns
    -------
    numpy.ndarray
        Element in reference frame of center of mass.
        The shape remains the same as the input array.
    """
    center_of_mass = np.nanmean(segmentation_element, axis=1)

    dim = segmentation_element.ndim

    if dim == 3:
        centroids = segmentation_element - center_of_mass[:, None, :]
        return centroids
    elif dim == 4:
        boxes = segmentation_element - center_of_mass[:, None, None, :]
        return boxes


def subtract_center_of_mass_seg_dict(seg):
    """
    Subtract the center of mass from centroids and bounding boxes in a segmentation dictionary.

    Parameters
    ----------
    seg : dict
        The segmentation dictionary containing centroids and bounding boxes.

    Returns
    -------
    dict
        A deepcopy of the input segmentation dictionary with the center of mass subtracted from centroids
        and bounding boxes.
    """
    seg_reref = deepcopy(seg)
    center_mass = np.nanmean(seg["centroids"], axis=1)
    seg_reref["centroids"] = seg["centroids"] - center_mass[:, None, :]
    seg_reref["bounding_boxes"] = seg["bounding_boxes"] - center_mass[:, None, None, :]
    return seg_reref


def filter_seg_dict(seg, appearance_threshold=300, verbose=False):
    """
    Filter a segmentation dictionary based on appearance threshold.

    Parameters
    ----------
    seg : dict
        The segmentation dictionary containing centroids, bounding boxes, orientations, and confidence values.

    appearance_threshold : int, optional
        The minimum number of appearances for a segment to be retained.
        Segments appearing fewer times than this threshold will be removed. Default is 300.

    Returns
    -------
    dict
        A deepcopy of the input segmentation dictionary with segments filtered based on appearance threshold.
    """
    
    whale_frame_counts = np.sum(~np.isnan(seg["orientations_rad"]), axis=0)
    mask_keep_idx = whale_frame_counts > appearance_threshold

    idx_to_remove = np.where(~mask_keep_idx)[0]

    filtered_seg = deepcopy(seg)
    filtered_seg["bounding_boxes"] = seg["bounding_boxes"][:, mask_keep_idx]
    filtered_seg["centroids"] = seg["centroids"][:, mask_keep_idx]
    filtered_seg["orientations_rad"] = seg["orientations_rad"][:, mask_keep_idx]
    filtered_seg["orientations_confidence"] = seg["orientations_confidence"][:, mask_keep_idx]

    if "masks_arrays" in seg.keys():
        filtered_seg["masks_arrays"] = seg["masks_arrays"][:, mask_keep_idx]
    if "masks_polygons" in seg.keys():
        filtered_seg["masks_polygons"] = seg["masks_polygons"][:, mask_keep_idx]

    
    filtered_seg["num_whales"] = filtered_seg["orientations_rad"].shape[1]
    filtered_seg["num_whales_over_time"] = num_whales_over_time(filtered_seg["orientations_rad"])

    filtered_seg["ids"] = list(np.delete(seg["ids"], idx_to_remove, axis=0))
    filtered_seg["id_numbers"] = list(np.delete(seg["id_numbers"], idx_to_remove, axis=0))

    if verbose:
        print(f"Number of whales reduced from {seg['num_whales']} to {seg['num_whales']}")
        print(f"Whale indices removed: {idx_to_remove}")
        ids_removed = [i for i in seg["ids"] if i not in filtered_seg["ids"]]
        print(f"Whale IDs removed: {ids_removed}")

    return filtered_seg


def remove_nonsegmented_frames(seg):
    """
    Remove frames that have no segmented data from the segment dictionary.

    This function identifies frames where all orientation values are NaN and removes these frames
    from all relevant arrays in the segment dictionary.

    Parameters
    ----------
    seg : dict
        Segmentation dictionary

    Returns
    -------
    seg_new : dict
        A new dictionary with the same structure as `seg`, but with frames containing only NaN values removed.
    """
    
    #orientations = seg["orientations_rad"]
    #confidence = seg["orientations_confidence"]
    #bboxes = seg["bounding_boxes"]
    #centroids = seg["centroids"]
    
    # nan_mask = np.isnan(orientations)
    # all_nan_times = np.all(nan_mask, axis=1)
    # nan_times_indices = np.where(all_nan_times)[0]

    keep_indices = seg["are_frames_segmented"]

    # orientations_new = seg["orientations_rad"][keep_indices] #np.delete(orientations, nan_times_indices, axis=0)
    # confidence_new = seg["orientations_confidence"][keep_indices] #.delete(confidence, nan_times_indices, axis=0)
    # bboxes_new = seg["bounding_boxes"][keep_indices] #np.delete(bboxes, nan_times_indices, axis=0)
    # centroids_new = seg["centroids"][keep_indices] # np.delete(centroids, nan_times_indices, axis=0)
    # if "masks_polygon" in seg.keys():
    #     masks_new = seg["masks_polygon"][keep_indices] # np.delete(seg["masks_polygon"], nan_times_indices, axis=0)
    
    seg_new = deepcopy(seg)
    
    seg_new["orientations_rad"] = seg["orientations_rad"][keep_indices]
    seg_new["orientations_confidence"] = seg["orientations_confidence"][keep_indices]
    seg_new["bounding_boxes"] = seg["bounding_boxes"][keep_indices]
    seg_new["centroids"] = seg["centroids"][keep_indices]
    if "masks_polygon" in seg.keys():
        seg_new["masks_polygon"] = seg["masks_polygon"][keep_indices]

    seg_new["frame_indices"] = seg["frame_indices"][keep_indices]
    seg_new["timestamps_s"] = seg["timestamps_s"][keep_indices]
    seg_new["timestamps_s_babyTime"] = seg["timestamps_s_babyTime"][keep_indices]
    seg_new["timestamps_str"] = seg["timestamps_str"][keep_indices]
    seg_new["are_frames_segmented"] = seg["are_frames_segmented"][keep_indices] 
    seg_new["num_whales_over_time"] = seg["num_whales_over_time"][keep_indices]

    seg_new["num_frames"] = sum(keep_indices)
    
    return seg_new


def slice_seg(seg, frame_min=None, frame_max=None):
    """
    Remove frames that before frame_min and after frame_max

    Parameters
    ----------
    seg : dict
        Segmentation dictionary

    Returns
    -------
    seg_new : dict
        A new dictionary with the same structure as `seg`
    """
    
    orientations = seg["orientations_rad"]
    confidence = seg["orientations_confidence"]
    bboxes = seg["bounding_boxes"]
    centroids = seg["centroids"]
    
    seg_new = deepcopy(seg)
    
    seg_new["orientations_rad"] = orientations[frame_min:frame_max]
    seg_new["orientations_confidence"] = confidence[frame_min:frame_max]
    seg_new["bounding_boxes"] = bboxes[frame_min:frame_max]
    seg_new["centroids"] = centroids[frame_min:frame_max]
    if "masks_polygon" in seg.keys():
        seg_new["masks_polygon"] = masks[frame_min:frame_max]

    seg_new["frame_indices"] = seg["frame_indices"][frame_min:frame_max]
    seg_new["timestamps_s"] = seg["timestamps_s"][frame_min:frame_max]
    seg_new["timestamps_s_babyTime"] = seg["timestamps_s_babyTime"][frame_min:frame_max]
    seg_new["timestamps_str"] = seg["timestamps_str"][frame_min:frame_max]
    seg_new["are_frames_segmented"] = seg["are_frames_segmented"][frame_min:frame_max]

    seg_new["num_frames"] = len(seg_new["frame_indices"])
    seg_new["num_whales_over_time"] = seg["num_whales_over_time"][frame_min:frame_max]
    seg_new["total_time"] = (seg_new["num_frames"] / seg["fps"]) / 60 # in minutes

    return seg_new


def add_masks_polygons(seg):
    """
    Compute and add masks in Polygon format to the segmentation `seg`.

    Parameters
    ----------
    seg : dict
        Segmentation dictionary

    Returns
    -------
    seg : dict
        Input dictionary with the masks in polygon format

    """

    if "masks_polygons" in seg.keys():
        print("Replacing already existing masks polygons")


    num_whales = seg["num_whales"]
    num_frames = seg["num_frames"]
    contour = 0

    # compute the polygons
    
    masks_arrays = seg["masks_arrays"]
    masks_polygons = np.empty((num_frames, num_whales), dtype=object)


    for t in range(num_frames):
        
        for whale_idx in range(num_whales):
            
            # remove invalid values
            coords = [i for i in masks_arrays[t, whale_idx, contour] if not np.all(i==-1)]
            
            mask_polygon = Polygon(coords)

            masks_polygons[t, whale_idx] = mask_polygon

    seg["masks_polygons"] = masks_polygons

    return seg


            
