import os

import h5py
import numpy as np

from utils_networks import num_whales_over_time

__all__ = ["check_s3_folder", "fetch_data", "load_file"]


def check_s3_folder(s3folder):
    os.system(f'aws s3 ls {s3folder} --recursive | grep ".hdf5$"')


def fetch_data(videos, s3folder, drone, destination_folder):
    for video in videos:
        source_path = f"{s3folder}{drone}{video}/{video}_segmentations.hdf5"
        destination_path = f"{destination_folder}/{video}_segmentations.hdf5"
        os.system(f"aws s3 cp {source_path} {destination_path}")


def load_file(number, folder):
    path = f"../data/{folder}/{number}_segmentations.hdf5"
    file = h5py.File(path, "r")
    return file

def convert_to_dict(segmentations, frame_times=None, with_masks=True):

        num_frames = segmentations.get_num_frames_total()
        num_whales = segmentations.get_num_whales()
        frames_indices = np.arange(num_frames)
        fps = 30 # number of frames per second in original video

        if frame_times is not None:
            timestamps_s, timestamps_s_babyTime, timestamps_str = frame_times
        else: 
            timestamps_s, timestamps_s_babyTime, timestamps_str = None, None, None

        # deal with different in +-1 in number of frames and timestamps
        #if len(timestamps_s) != num_frames:
        num_frames_new = min(num_frames, len(timestamps_s))

        # Specify which bounding box type to use in the below examples.
        # Can be 'full', 'head', or 'tail'
        bounding_box_keys = segmentations.get_bounding_box_keys() # get the available types for reference
        bounding_box_key = 'full'

        # Get all bounding boxes of the desired type.
        # Will return an h5py Dataset instead of loading it into memory.
        bounding_boxes_4xy = segmentations.get_all_bounding_boxes_4xy(bounding_box_key=bounding_box_key)
        # Can cast the matrix (or a slice of it) to a numpy array to load it into memory.
        bounding_boxes_4xy = np.array(bounding_boxes_4xy)
        bounding_boxes_4xy_reshaped = bounding_boxes_4xy.reshape((num_frames, num_whales, 4, 2)) # if desired, can reshape to have each box corner as a matrix row

        # Get all centroids.
        centroids_xy = segmentations.get_all_centroids_xy()

        # Get all orientations with confidences.
        orientations_rad_confidence = segmentations.get_all_orientations_rad_confidence()
        orientations_rad = orientations_rad_confidence[:, :, 0]
        orientations_confidence = orientations_rad_confidence[:, :, 1]

        # convert from hdf5 to numpy array 

        bounding_boxes_4xy_reshaped = np.array(bounding_boxes_4xy_reshaped)
        centroids_xy = np.array(centroids_xy)
        orientations_rad = np.array(orientations_rad)
        orientations_confidence = np.array(orientations_confidence)

        if with_masks:
            # get masks and convert to array (may take a minute)
            print("Loading and converting masks, this may take a minute..")
            masks_raw = np.array(segmentations.get_all_masks_contours()) 
            print("Done")
            #masks_raw = segmentations.get_all_masks_contours()

        # deal with frames and sampling frequencies
        frame_segmented = segmentations.get_frames_are_segmented()
        are_frames_segmented = frame_segmented[:,0].astype(bool)

        # distance (in frames) between two segmented frames)
        segmentation_frequency = set(np.diff(frames_indices[are_frames_segmented]))
        segmentation_frequency = [i for i in segmentation_frequency if i < 90]
        if len(segmentation_frequency) > 1:
            print(f"There should be only one segmentation frequency. segmentation_frequency is {segmentation_frequency}")
        
        #segmentation_frequency = list(segmentation_frequency)[0]
        
        # format IDs
        ids = segmentations.get_whale_ids()
        ids = [el.split(" (")[0] for el in ids]
        ids = [el if "AutoMerged" not in el else el[-5:] for el in ids]
        ids = [f"Whale {i}" if el=="" else el for i, el in enumerate(ids)]
        ids = ["Snow" if el=="SNOW" else el for el in ids]
        ids = ["Newborn" if el=="newborn" else el for el in ids]

        # store all
        seg = dict()

        # can be modified later
        seg["bounding_boxes"] = bounding_boxes_4xy_reshaped[:num_frames_new]
        seg["centroids"] = centroids_xy[:num_frames_new]
        seg["orientations_rad"] = orientations_rad[:num_frames_new]
        seg["orientations_confidence"] = orientations_confidence[:num_frames_new]

        if with_masks:
            seg["masks_arrays"] = masks_raw[:num_frames_new]

        seg["frame_indices"] = frames_indices[:num_frames_new]
        seg["timestamps_s"] = np.array(timestamps_s)[:num_frames_new]
        seg["timestamps_s_babyTime"] = np.array(timestamps_s_babyTime)[:num_frames_new]
        seg["timestamps_str"] = np.array(timestamps_str)[:num_frames_new]

        seg["num_whales"] = num_whales
        seg["num_frames"] = num_frames_new
        seg["num_whales_over_time"] = num_whales_over_time(orientations_rad)[:num_frames_new]

        # won't be modified later
        seg["ids"] = ids
        seg["id_numbers"] = segmentations.get_whale_id_numbers()
        seg["are_frames_segmented"] = are_frames_segmented[:num_frames_new]
        seg["fps"] = fps
        seg["segmentation_frequency"] = segmentation_frequency
        seg["total_time"] = (num_frames_new / fps) / 60 # in minutes

        return seg
