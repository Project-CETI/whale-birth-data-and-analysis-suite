import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from utils_networks import *


def compute_vector_angle(a):
    if len(a) != 2:
        raise IndexError("vector a expected to be length 2")        
    x = a[0]
    y = a[1]
    rad = np.arctan2(y, x)
    rad = np.mod(rad, 2*np.pi)
    return rad


def compute_angles_to_whale_target(seg, target_name):   
    
    #create a new copy of seg
    seg_filtered = copy.deepcopy(seg)
 
    #find idx of the target
    target_idx = seg_filtered['ids'].index(target_name)
        
    #Calcuate indeces of whales touching calf
    #adj_box_overlap = build_adjacency(seg_filtered, box_overlap_binary)
    
    # set diagonals to 0 again
    #for i in range(adj_box_overlap.shape[0]):
        #np.fill_diagonal(adj_box_overlap[i, :, :], 0)
    adj_box_overlap = seg_filtered['adj_box_overlap']

    #remove whales that are touching the target
    n_frame = len(seg_filtered['centroids'])
    
    for i in range(n_frame):
        target_touching = adj_box_overlap[i, target_idx]
        target_touching_idxs = np.where(target_touching)
        seg_filtered['centroids'][i][target_touching_idxs] = np.nan
        seg_filtered['orientations_rad'][i][target_touching_idxs] = np.nan
        seg_filtered['bounding_boxes'][target_touching_idxs] = np.nan
        
    #Subtract the target's centroid from all the centroids
    target_centroids = seg_filtered['centroids'][:,target_idx].copy()
    
    #compute vector to target
    n_whales = seg_filtered['centroids'].shape[1]
    seg_filtered['vector_to_target'] = seg_filtered['centroids'].copy()
    for n in range(n_whales):    
        seg_filtered['vector_to_target'][:,n, 0] = target_centroids[:,0] - seg_filtered['centroids'][:,n,0]
        seg_filtered['vector_to_target'][:,n, 1] = seg_filtered['centroids'][:,n,1] - target_centroids[:,1]
        
    #compute orientation and position angle and difference between them
    orientation_whales_target = np.apply_along_axis(compute_vector_angle, 2, seg_filtered['vector_to_target'])
    orientations_whales = np.mod(seg_filtered['orientations_rad'], 2*np.pi)
    angles_target_sightline = np.mod(orientation_whales_target - orientations_whales, 2*np.pi)
    
    #remove angle target with itself
    angles_target_sightline[:,target_idx] = np.nan

    return angles_target_sightline

def plot_whales_average_attention(angles_dict, video_name, save=False):
    
    whales = angles_dict.keys()
    
    num_whales = len(whales)
    colors = sb.color_palette("tab20", n_colors=num_whales)

    x_ticks = whales
    
    whales_mean_attention = []
    whales_std_attention = []
    
    for whale in whales:
        mean = np.nanmean(np.nanmean(np.cos(angles_dict[whale]), axis=0))
        std = np.nanstd(np.nanmean(np.cos(angles_dict[whale]), axis=0))
        whales_mean_attention.append(mean)
        whales_std_attention.append(std)
        
    plt.figure()
    
    plt.bar(x=range(num_whales), height=whales_mean_attention, color=colors)
    plt.errorbar(range(num_whales), whales_mean_attention, yerr=whales_std_attention, fmt="o", color="k")

    plt.title(f"Avarage attention from others - {video_name}")
    plt.xlabel("Whale ids")
    plt.xticks(range(num_whales), x_ticks);
    plt.xticks(rotation=45);
    plt.tight_layout()

    if save:
        plt.savefig(f"../results/attention/whales_means_{video_name}.png", dpi=300, bbox_inches='tight')
        plt.subplots_adjust(bottom=0.2)
