import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import argparse
#import ultralytics
#ultralytics.checks()
#from ultralytics import YOLO
import copy
from skimage.measure import regionprops
import sys
from Segmentations import Segmentations
#########################################
#from DINO_FEATURES.collect_dino_features import *
#from DINO_FEATURES.dino_wrapper import *
from collections import OrderedDict
import matplotlib
cmap = matplotlib.cm.get_cmap("jet")
import math
cosine_similarity = torch.nn.CosineSimilarity(dim=1) 
# Initial parser for the input video name
initial_parser = argparse.ArgumentParser(add_help=False)
initial_parser.add_argument('--input_video_name', type=str, default='1688829190202')
initial_parser.add_argument('--min_mask_size', type=float, default='2000')
initial_args, unknown = initial_parser.parse_known_args()

# Main parser with all arguments
parser = argparse.ArgumentParser(description='')

parser.add_argument('--input_video_name', 
                        type=str, default=initial_args.input_video_name ,
                        help='video name to process') #1688830961248

parser.add_argument('--input_video_path', 
                     type=str, metavar='PATH',
                     default= f'/data/vision/data/raw_videos/2023-07-08/CETI-DJI_MAVIC3-1/{initial_args.input_video_name}.MP4',
                     help='path to video')

parser.add_argument('--folder_to_store', 
                     type=str, default= f'/data/vision/data/tracking_results/final_outputs_combined/{initial_args.input_video_name}',
                     )


parser.add_argument('--seg_path', 
                     type=str, default= f'/data/vision/data/results/all_fastsam_split_augmented_imgsize_1024_{initial_args.input_video_name}/{initial_args.input_video_name}/best_checkpoint',
                     )


parser.add_argument('--queries_dir', 
                     type=str, default= 'stored_queries/save',
                     )

parser.add_argument('--load_from', 
                     type=str, default= 'tracking',
                     )


## add argument for the finetuned model path
parser.add_argument('--finetuned_model_path', default=f'/data/vision/data/runs/segment/all_fastsam_split_augmented_imgsize_1024_{initial_args.input_video_name}/weights', type=str, help='path to finetuned model weights')

## add argument for resuming from a specific frame index (default is 0)
parser.add_argument('--resume_from', default=0, type=int, help='frame index to resume from') 
parser.add_argument('--stop_at', default=0, type=int, help='frame index to end at') 
#parser.add_argument('--use_dino', default=0, type=int, help='') 
#### Hyper parameters
parser.add_argument('--min_mask_size', 
                     type=float, default= initial_args.min_mask_size,
                   )

parser.add_argument('--min_certainty', 
                     type=float, default= '0.93',
                   )

parser.add_argument('--miou_thresh',
                     type=float, default= '0.25',
                   )

parser.add_argument('--max_tracking_duration', 
                     type=float, default= '100',
                   )

parser.add_argument('--max_memory', 
                     type=float, default= '200',
                   )

parser.add_argument('--use_different_boxes', 
                     type=int, default= 1,
                   )
args = parser.parse_args()



def plot_similarity_if_neded(frame, similarity_rel, alpha = 0.5):
    
        img_to_viz = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_to_viz = cv2.resize(img_to_viz, (similarity_rel.shape[-1], similarity_rel.shape[-2]))
        similarity_colormap = cmap(similarity_rel)[..., :3]
        
        _overlay = img_to_viz.astype(np.float32) / 255
        _overlay = (1-alpha) * _overlay + (alpha) * similarity_colormap
        _overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)
        return _overlay
        
def multiclass_vis(class_labels, img_to_viz, num_of_labels, np_used = False,alpha = 0.5):
    _overlay = img_to_viz.astype(float) / 255.0
    if np_used:
         viz = cmap(class_labels/num_of_labels)[..., :3]
    else:
         class_labels = class_labels.detach().cpu().numpy().astype(float)
         viz = cmap((class_labels/num_of_labels))[..., :3]
    _overlay =  alpha * viz + (1-alpha) * _overlay 
    s_overlay = cv2.cvtColor(np.float32(_overlay), cv2.COLOR_BGR2RGB)  

    return _overlay

def get_queries(cfg):
    queries = OrderedDict({})    
    for file_name in os.listdir(cfg['queries_dir']):
        if file_name.startswith("feat") and file_name.endswith(".pt"):
            full_path = "{}/{}".format(cfg['queries_dir'], file_name)
            query = torch.load(full_path) 
            if not isinstance(query, list): # annotations
                query = [query]
            key = file_name[4:-3]
            queries[key] = query
            
    if not queries.keys(): 
            print("No annotations found in {}!!!!, see step 1 and script annotate_features.py".format(cfg['queries_dir']))
            exit("1")

    mean_queries = OrderedDict({})
    for key,query in queries.items():
        query = torch.stack(query).cuda().mean(dim=0)
        query = torch.nn.functional.normalize(query, dim=0)
        mean_queries[key] = query
    return mean_queries
        #else:
        #    return queries
        #if cfg['metric'] == 'closest_mean':



def colorize_mask(pred_mask):

    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):

    img_mask = np.zeros_like(img)
    img_mask = img
    
    if id_countour:

        # very slow ~ 1s per image

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]
        for id in obj_ids:
            # Overlay color on  binary mask

            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]

            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0

    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0

    return img_mask.astype(img.dtype)




def get_mask_size(mask, label):
    return (mask == label).sum()

def is_label_in_dict(label,dict_to_check):
    return label in dict_to_check.keys()

def update_all_seen_whales(masks, frame_idx):
    current_labels =  np.unique(masks)
    for label in current_labels:
        if not is_label_in_dict(label,all_seen_whales):
            all_seen_whales[label] = {}
        all_seen_whales[label]["mask"] = masks == label
        all_seen_whales[label]["last_seen"] =  frame_idx

def binaryMaskIOU(mask1, mask2, return_all = False):
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1,  mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)
    if return_all:
        return iou, intersection, mask1_area+mask2_area-intersection
    else:
        return iou     

def get_max_key_value_in_dicts(dict_to_parse):
   
    max_ = 0
    used_key = None
    for key, val in dict_to_parse.items():
        if max_ < val: 
            max_ = val
            used_key = key
    return used_key, max_

def align(seg_mask, prev_perd_masks, new_tracked_mask, pixel_cerntainty, frame_idx):
    global max_whale_idx

    iou_dict = {}
    current_seg_labels  =  np.unique(seg_mask)
    prev_tracked_labels =  np.unique(prev_perd_masks)
    
    #compute iou between every label in current_seg_labels and prev_tracked_labels
    for seg_label in current_seg_labels:
        iou_dict[seg_label] = {}
        for tracked_label in prev_tracked_labels:
            iou_dict[seg_label][tracked_label] = binaryMaskIOU(seg_mask == seg_label, prev_perd_masks == tracked_label)
        for seen_whale_label in all_seen_whales.keys():
            since_last_seen = frame_idx - all_seen_whales[seen_whale_label]['last_seen'] 
            if not seen_whale_label in iou_dict[seg_label].keys() and since_last_seen< args.max_memory:
                iou_dict[seg_label][seen_whale_label] = binaryMaskIOU(seg_mask == seg_label, all_seen_whales[seen_whale_label]['mask'])

    pred_mask = np.zeros(seg_mask.shape)

    #Handle new segmented whales
    for seg_label in current_seg_labels:
        closest_tracked_whale, max_iou = get_max_key_value_in_dicts(iou_dict[seg_label])
        if get_mask_size(seg_mask, seg_label) < args.min_mask_size: continue 
        if max_iou > args.miou_thresh:
            pred_mask[seg_mask==seg_label] = closest_tracked_whale
        else: 
            max_whale_idx +=1
            pred_mask[seg_mask==seg_label] = max_whale_idx

    pred_by_seg = copy.deepcopy(pred_mask)
    pred_mask_labels =  np.unique(pred_mask)
    for tracked_label in prev_tracked_labels:
        mask_certainty  = pixel_cerntainty[new_tracked_mask == tracked_label].mean()
        mask_size       = get_mask_size(new_tracked_mask, tracked_label)
        since_last_seen = frame_idx - all_seen_whales[tracked_label]['last_seen'] 
        
        try :
            print("tracked mask size=", get_mask_size(new_tracked_mask, tracked_label), "pixels with low_certainty= ", (pixel_cerntainty[new_tracked_mask == tracked_label]<0.9).sum())
            print("mean cetaitny= ", pixel_cerntainty[new_tracked_mask == tracked_label].mean(),"min certainty=", pixel_cerntainty[new_tracked_mask == tracked_label].min()) 
        
        except:
            print("error in printing mask size and certainty")
       
        if  mask_size < args.min_mask_size or mask_certainty < args.min_certainty or since_last_seen >args.max_tracking_duration: continue
        print("passed mask_checked")
        if not tracked_label in pred_mask_labels:
            print("used_tracking")
            pred_mask[np.logical_and(new_tracked_mask==tracked_label,pred_mask ==0)] = tracked_label
        print("------------------------------------------------------------------------------------------------------------------------------------------")
    
    return pred_mask, pred_by_seg

def split_box_into_2(box):
    base, left_up, head, right_up = box
    base, left_up, head, right_up = np.array(base), np.array(left_up), np.array(head), np.array(right_up)
    
    width = np.linalg.norm(base - left_up)
    height = np.linalg.norm(base -right_up)

    if height> width:
        box_1_right_up = (right_up + base)/2
        box_1_base = base
        box_1_left_up = left_up
        box_1_head = (head + left_up)/2

        
        box_2_base = (right_up + base)/2
        box_2_right_up = right_up
        box_2_left_up = (head + left_up)/2
        box_2_head = head
    else:
        box_1_base = base
        box_1_right_up = right_up
        box_1_left_up = (left_up + base)/2
        box_1_head = (right_up + head)/2

        box_2_base = (base + left_up)/2
        box_2_right_up = (right_up + head)/2
        box_2_left_up = left_up
        box_2_head = head

    box_1 = [box_1_base, box_1_left_up, box_1_head, box_1_right_up]
    box_2 = [box_2_base, box_2_left_up, box_2_head, box_2_right_up]
    return box_1, box_2
def calc_orientation(mask, use_different_boxes, props):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_cnt = -1
        for i,cnt in enumerate(contours):
            if max_cnt < len(cnt): 
                max_cnt = len(cnt)
                cnt_idx = i

        cnt = contours[cnt_idx]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        

        box_1, box_2 = split_box_into_2(box) #box 1 is always down
        box_1, box_2 =  np.int0(box_1), np.int0(box_2)
        


        box_1_mask =  np.zeros(mask.shape, dtype=np.uint8)
        box_1_mask = cv2.fillPoly(box_1_mask,[box_1],1)
        box_2_mask =  np.zeros(mask.shape, dtype=np.uint8)
        box_2_mask = cv2.fillPoly(box_2_mask,[box_2],1)

        if use_different_boxes:
            part_1_mask = np.logical_and(box_1_mask==1,  mask==1)
            part_2_mask = np.logical_and(box_2_mask==1,  mask==1)
            
            contours1, _ = cv2.findContours(part_1_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(part_2_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt1 = contours1[0]
            rect1 = cv2.minAreaRect(cnt1)
            box1 = cv2.boxPoints(rect1)
            box1 = np.int0(box1)

            cnt2 = contours2[0]
            rect2 = cv2.minAreaRect(cnt2)
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)

            box_1_mask =  np.zeros(mask.shape, dtype=np.uint8)
            box_1_mask = cv2.fillPoly(box_1_mask,[box1],1)
            box_2_mask =  np.zeros(mask.shape, dtype=np.uint8)
            box_2_mask = cv2.fillPoly(box_2_mask,[box2],1)

            
            
        else:
           
            box1 = box_1
            box2 = box_2
        #cv2.imwrite(os.path.join(output_dir,"colored", "{}_head_tail.jpg".format(str(frame_idx))), masked_frame_saved)

        _, intersection_1, _  = binaryMaskIOU(box_1_mask, mask, return_all = True)
        _, intersection_2, _  = binaryMaskIOU(box_2_mask, mask, return_all = True)

        miou_1, miou_2 = intersection_1/box_1_mask.sum(), intersection_2/box_2_mask.sum()
        certainty = abs(miou_1 - miou_2)/min(miou_2,miou_1)

        decide_via_up_down = True
        if box_1[0][-1]<box_2[0][-1]:
            up_box_miou = miou_1
            down_box_mio = miou_2
            up_box_coordinates = box1
            down_box_coordinates = box2
        elif box_1[0][-1]>box_2[0][-1]:
            up_box_miou = miou_2
            down_box_mio = miou_1
            up_box_coordinates = box2
            down_box_coordinates = box1
        else:
            decide_via_up_down = False

        decide_via_right_left = True
        if box_1[0][0]< box_2[0][0]:
                left_box_miou = miou_1
                right_box_miou = miou_2
                left_box_coordinates = box1
                right_box_coordinates = box2
        elif box_1[0][0] > box_2[0][0]:
                left_box_miou = miou_2
                right_box_miou = miou_1
                left_box_coordinates = box2
                right_box_coordinates = box1
        else:
            decide_via_right_left = False
        
        if decide_via_up_down: 
            if down_box_mio>= up_box_miou:
                head_box_up_down = "down"
                head_box_coordinates = down_box_coordinates
                tail_box_coordinates = up_box_coordinates
            else:
                head_box_up_down = "up"
                head_box_coordinates = up_box_coordinates
                tail_box_coordinates = down_box_coordinates
        else:
            head_box_up_down = "nutral"

        if decide_via_right_left:
            if left_box_miou>= right_box_miou:
                head_box_right_left = "left"
                head_box_coordinates = left_box_coordinates
                tail_box_coordinates = right_box_coordinates
            else:
                head_box_right_left = "right"
                head_box_coordinates = right_box_coordinates
                tail_box_coordinates = left_box_coordinates
        else:
            head_box_right_left =  "nutral"
       
        angle_in_degrees_bfeore = props.orientation * (180/np.pi) + 90 
        
        #input()
        
        if abs(angle_in_degrees_bfeore) < 90 and abs(angle_in_degrees_bfeore)>0:
            if ("up" in head_box_up_down and "right" in head_box_right_left) or ("up" in head_box_up_down and "nutral" in head_box_right_left) or ("nutral" in head_box_up_down and "right" in head_box_right_left):
                angle_in_degrees =  angle_in_degrees_bfeore
            else:
                angle_in_degrees = angle_in_degrees_bfeore + 180
        elif abs(angle_in_degrees_bfeore) == 90:
            if "up" in head_box_up_down:
                angle_in_degrees =  angle_in_degrees_bfeore
            else:
                angle_in_degrees = angle_in_degrees_bfeore + 180
        elif abs(angle_in_degrees_bfeore) == 0:
            if "right" in head_box_right_left:
                angle_in_degrees =  angle_in_degrees_bfeore
            else:
                angle_in_degrees = angle_in_degrees_bfeore + 180
        else:
            if ("up" in head_box_up_down and "left" in head_box_right_left) or ("up" in head_box_up_down and "nutral" in head_box_right_left) or ("nutral" in head_box_up_down and "left" in head_box_right_left):
               angle_in_degrees =  angle_in_degrees_bfeore
            else:
                angle_in_degrees = angle_in_degrees_bfeore + 180

        
        
        return angle_in_degrees, certainty, box, head_box_coordinates, tail_box_coordinates 
####################################INIT TRACKER args#######################################
sam_args['generator_args'] = {
        'points_per_side': 30,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    }
segtracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
}


torch.cuda.empty_cache()
gc.collect()
segtracker = SegTracker(segtracker_args,sam_args,aot_args)
segtracker.restart_tracker()
##################################INIT DINO###############################################
Width     = 2500 #1250#2500  #int(3840/2) 1407,2500
Height    = 1407 #704#1407  #int(2160/2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


##################################INIT relevant directories###############################
# output masks
output_dir = args.folder_to_store
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'colored')): os.mkdir(os.path.join(output_dir, 'colored'))
if not os.path.exists(os.path.join(output_dir, 'orig')): os.mkdir(os.path.join(output_dir, 'orig'))
if not os.path.exists(os.path.join(output_dir, 'np_masks')): os.mkdir(os.path.join(output_dir, 'np_masks'))
if not os.path.exists(os.path.join(output_dir, 'masks')): os.mkdir(os.path.join(output_dir, 'masks'))

##################################INIT for HDF5###############################
#    query0 = torch.load("feat0.pt")  # Water annotations
#    query1 = torch.load("feat1.pt")  # Head annotations
#    query2 = torch.load("feat2.pt")  # Body annotations
#    query3 = torch.load("feat3.pt")  # Fluke annotations
#    queries = [query0, query1, query2, query3]
## create a var for hdf5 file path
hdf5_file_path = os.path.join(output_dir, f'{args.input_video_name}_segmentations.hdf5')


# video file paths
video_filepaths = { # the keys can be anything you want
                    'masks_boxes_vectors': os.path.join(output_dir, '%s_segmentations_boundingBoxes_orientations.mp4' % args.input_video_name),
                    'only_masks':         os.path.join(output_dir,  '%s_segmentations.mp4' % args.input_video_name),
                    'masks_boxes_head_fluke':  os.path.join(output_dir,  '%s_segmentations_boundingBoxes_head_fluke.mp4' % args.input_video_name)
                }

# Remove existing outputs.
if os.path.exists(hdf5_file_path): os.remove(hdf5_file_path)
  
for video_filepath in video_filepaths.values():
  if os.path.exists(video_filepath): os.remove(video_filepath)

##################################INITs for the script###############################




mask_shape = (Height, Width)
MAX_POSSIBLE_WHALE_INDEX = 200 # 9


## create segmentation object
segmentations = Segmentations(h5_filepath= hdf5_file_path,
                video_filepaths=video_filepaths,
                mask_shape=mask_shape,
                max_whale_index=MAX_POSSIBLE_WHALE_INDEX,
                video_fps=30)


cap = cv2.VideoCapture(args.input_video_path)
npy_seg_dir = os.path.join(args.seg_path, "masks_npy")
vis_seg_dir = os.path.join(args.seg_path, "masks")

all_seen_whales = {}
global max_whale_idx

max_whale_idx = 1
frame_idx     = args.resume_from
if args.resume_from>0:  cap.set(cv2.CAP_PROP_POS_FRAMES,args.resume_from -1)

#########################################################################################################
use_different_boxes = args.use_different_boxes
with torch.cuda.amp.autocast():
    while True:

        ret, frame = cap.read()##file_name = os.path.join(seg_path,'orig', )#frame = cv2.imread(filename)
        if not ret: break
        if args.stop_at >0 and frame_idx > args.stop_at: break

        frame = cv2.resize(frame, (Width,Height))#Remove  
        
        #cv2.imwrite(os.path.join(output_dir,"orig", "{}.jpg".format(str(frame_idx))), frame); frame_idx+=1;       continue
        vis_file_path = os.path.join(vis_seg_dir, f"frame_{frame_idx:06d}.jpg.png")      
        npy_file_path = os.path.join(npy_seg_dir, f"frame_{frame_idx:06d}.jpg.npy")   
        print("Parsing file:", vis_file_path)

        torch.cuda.empty_cache()
        gc.collect()
        
        if  frame_idx == args.resume_from:
            if args.resume_from > 0: 
                if args.load_from == 'tracking': npy_file_path = os.path.join(output_dir, 'np_masks' ,"{}.npy".format(str(frame_idx)))
                
            segtracker.restart_tracker()

            
            seg_masks = np.load(npy_file_path)
            pred_mask = seg_masks.astype(int)
     
            segtracker.add_reference(frame, pred_mask)
            update_all_seen_whales(pred_mask, frame_idx)
            max_whale_idx = np.max(pred_mask)
            
        elif os.path.exists(npy_file_path):  ## os.path.exists(vis_file_path) and   NOTE: updated this line
            
            seg_masks = np.load(npy_file_path).astype(int)
            
            new_tracked_mask, pixel_cerntainty = segtracker.track(frame, update_memory=True)
            new_tracked_mask = new_tracked_mask.astype(int)

            pred_mask, pred_by_seg = align(seg_masks, prev_perd_masks, new_tracked_mask, pixel_cerntainty, frame_idx)
            
            segtracker.restart_tracker()
            segtracker.add_reference(frame, pred_mask)
            update_all_seen_whales(pred_by_seg, frame_idx)

        else:
            pred_mask, pixel_cerntainty = segtracker.track(frame,update_memory=True)
        
        
        np.save(os.path.join(output_dir, 'np_masks' ,str(frame_idx)), pred_mask) # TODO might not need it in the future
        segmentations.add_mask(frame_index=frame_idx, mask_matrix=pred_mask) 
        
        masked_frame = draw_mask(frame, pred_mask) 
        # save the name with the format frame_000000.xxx by converting the index to string and padding it with zeros
        frame_idx_str = str(frame_idx).zfill(6)
        # save masks to videos
        segmentations.add_segmented_image( video_key='only_masks', 
                                           frame_index=frame_idx,
                                           img=masked_frame, 
                                           img_format='bgr',
                                           write_frame_index=True) ##TODO: verify that we don't need further conversion

        

        print("processed frame {} in tracking stage: obj_num {}".format(frame_idx, segtracker.get_obj_num()),end='\r')
        prev_perd_masks = copy.deepcopy(pred_mask)

       
        
        masked_frame_saved = copy.deepcopy(masked_frame)

        for idx in range(1, max_whale_idx+1):
            
            mask = pred_mask == idx
            mask = mask.astype("uint8")
            if mask.sum() == 0: continue
            
            #print(regionprops(mask.astype("uint8")), mask.sum())
            props = regionprops(mask.astype("uint8"))[0]#TODO
            
            ## add centroid
            segmentations.add_centroid(frame_index=frame_idx, whale_index=idx, centroid_yx=props.centroid)
            
            
            ##################################################################################################
            angle_in_degrees, certainty, box, head_box_coordinates, tail_box_coordinates = calc_orientation(mask, use_different_boxes, props)
           
            y0, x0 = props.centroid
            orientation =  (angle_in_degrees - 90)*(np.pi/180)  + np.pi / 2  # Rotate by 90 degrees

            arrow_length_multiplier = 2.0  # Adjust this value to change the arrow length
            x1 = int(x0 + np.cos(orientation) * arrow_length_multiplier * props.minor_axis_length)
            y1 = int(y0 - np.sin(orientation) * arrow_length_multiplier * props.minor_axis_length)

            segmentations.add_orientation(frame_index=frame_idx, whale_index=idx, orientation_rad=math.radians(angle_in_degrees), orientation_confidence=certainty)
            
            segmentations.add_bounding_box(bounding_box_key = 'full', frame_index=frame_idx, whale_index=idx, bounding_box_4xy = box.flatten())
            segmentations.add_bounding_box(bounding_box_key = 'head', frame_index=frame_idx, whale_index=idx, bounding_box_4xy = head_box_coordinates.flatten())
            segmentations.add_bounding_box(bounding_box_key = 'tail', frame_index=frame_idx, whale_index=idx, bounding_box_4xy = tail_box_coordinates.flatten())
            cv2.drawContours(masked_frame, [box], 0,(0,0,255),2)
            cv2.arrowedLine(masked_frame, (int(x0), int(y0)), (x1, y1), (255, 0, 255) , 3, tipLength=0.1)
            cv2.drawContours(masked_frame_saved, [head_box_coordinates], 0,(0,0,255),4)
            cv2.drawContours(masked_frame_saved, [tail_box_coordinates], 0,(0,255,255),4)
            cv2.arrowedLine(masked_frame_saved, (int(x0), int(y0)), (x1, y1), (255, 0, 255) , 3, tipLength=0.1)
        cv2.imwrite(os.path.join(output_dir,"colored", "{}.jpg".format(str(frame_idx))), masked_frame)
        segmentations.add_segmented_image(video_key='masks_boxes_vectors', 
                                          frame_index=frame_idx,
                                           img=masked_frame,   
                                           img_format='bgr',
                                           write_frame_index=True)

        segmentations.add_segmented_image(video_key='masks_boxes_head_fluke', 
                                          frame_index=frame_idx,
                                           img=masked_frame_saved,   
                                           img_format='bgr',
                                           write_frame_index=True)
        frame_idx += 1
        #masked_frame = cv2.resize(masked_frame, (1000,560))
        #cv2.imwrite(os.path.join(output_dir,"colored", "{}.jpg".format(str(frame_idx))), masked_frame)
        # if frame_idx == 20 :  break


    cap.release()

    ### 
    segmentations.close()

    print('\nfinished')







