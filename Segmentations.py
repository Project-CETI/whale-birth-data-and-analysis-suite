
############
#
# Copyright (c) 2023 Joseph DelPreto / MIT CSAIL and Project CETI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Created 2023 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

import numpy as np
import h5py
import os
import glob
from collections import OrderedDict

import ffmpeg
import cv2
import decord
from PIL import Image


###############################
# Helpers
###############################

def get_file_extension(filepath):
  if not isinstance(filepath, str):
    return None
  file_extension = os.path.splitext(filepath)[-1]
  file_extension = file_extension.lower()
  return file_extension

# Load an image from file.
# Optionally scale the image to a target size.
# Using the PIL method is fastest, since it can draft the downscaling during loading if it is a JPG.
# Will maintain the image's aspect ratio when scaling.
def load_image(filepath, target_width=None, target_height=None, method='pil'):
  img = None
  if method.lower() == 'opencv':
    img = cv2.imread(filepath)
    if target_width is not None and target_height is not None:
      img = scale_image(img, target_width=target_width, target_height=target_height)
  elif method.lower() == 'pil':
    img = Image.open(filepath)
    if target_width is not None and target_height is not None and get_file_extension(filepath) in ['jpg', 'jpeg']:
      img.draft('RGB', (int(target_width), int(target_height)))
      #print('target:', (target_width, target_height), '  drafted:', img.size, os.path.basename(filepath))
    try:
      img = np.asarray(img)
    except:
      return None
    if target_width is not None and target_height is not None:
      img = scale_image(img, target_width=target_width, target_height=target_height)
  return img

# Scale an image to fit within a target width and height.
# If maintaining the image's aspect ratio (which is the default),
#   Will scale to the largest size that fits within the target size.
# If the image size already meets the target criteria,
#   will return the original image (this call will not incur any delays beyond the size check).
# The input can be a numpy array or a PyQtGraph QPixmap object.
# Will also measure the total time spent in this method, for profiling purposes.
def scale_image(img, target_width, target_height, maintain_aspect_ratio=True):
  # Scale a numpy array.
  if isinstance(img, np.ndarray):
    img_width = img.shape[1]
    img_height = img.shape[0]
    # Determine an appropriate scale factor by considering both dimensions.
    scale_factor_byWidth = target_width/img_width if target_width is not None else None
    scale_factor_byHeight = target_height/img_height if target_height is not None else None
    # If maintaining the aspect ratio, use the same factor for both dimensions.
    if maintain_aspect_ratio:
      scale_factor = 1
      if scale_factor_byWidth is not None and scale_factor_byHeight is not None:
        scale_factor = min(scale_factor_byWidth, scale_factor_byHeight)
      elif scale_factor_byWidth is not None:
        scale_factor = scale_factor_byWidth
      elif scale_factor_byHeight is not None:
        scale_factor = scale_factor_byHeight
      if scale_factor != 1:
        res = cv2.resize(src=img, dsize=(0,0), fx=scale_factor, fy=scale_factor)
      else:
        # Do nothing if the image size is already as desired.
        res = img
      return res
    else:
      # If not maintaining the aspect ratio, scale each dimension by its computed factor.
      if scale_factor_byWidth != 1 or scale_factor_byHeight != 1:
        res = cv2.resize(src=img, dsize=(0,0), fx=scale_factor_byWidth, fy=scale_factor_byHeight)
      else:
        # Do nothing if the image size is already as desired.
        res = img
      return res
  else:
    raise ValueError('Unsupported image type for scaling')
    
# Draw text on an image, with a shaded background.
# pos is the target (x, y) position of the upper-left corner of the text, except:
#   If y is -1, will place the text at the bottom of the image.
#   If x is -1, will place the text at the right of the image.
#   If x and/or y is between 0 and 1, will center the text at that ratio of the width and/or height.
# If text_width_ratio is not None, will compute a font scale such that the text width is that fraction of the image width.
#   font_scale will be ignored if text_width_ratio is not None.
# If preview_only is True, will compute the text size but will not edit the image.
# The input image and any color arguments should be in BGR format, scaled out of 255.
def draw_text_on_image(img_bgr, text, pos=(0, 0),
                       font_scale=8, text_width_ratio=None,
                       font_thickness=1, font=cv2.FONT_HERSHEY_DUPLEX,
                       text_color_bgr=None,
                       text_bg_color_bgr=None, text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0.03,
                       preview_only=False,
                       ):
  # If desired, compute a font scale based on the target width ratio.
  if text_width_ratio is not None:
    if len(text) > 0:
      target_text_w = text_width_ratio * img_bgr.shape[1]
      font_scale = 0
      text_w = 0
      while text_w < target_text_w:
        font_scale += 0.2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
      font_scale -= 0.2
    else:
      font_scale = 1
  # Compute the text dimensions.
  (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  # Compute padding.
  text_bg_pad = round(text_w*text_bg_pad_width_ratio)
  text_bg_outline_width = round(text_w*0.02) if text_bg_outline_color_bgr is not None else 0
  # Compute the text position.
  # Place the text at the bottom and/or right if desired, and handle fractional placement if desired.
  x, y = pos
  if y == -1:
    y = round(img_bgr.shape[0] - text_h - text_bg_pad*2 - text_bg_outline_width*2)
  elif y > 0 and y < 1:
    y = round(img_bgr.shape[0]*y - text_h/2 - text_bg_pad - text_bg_outline_width)
  if x == -1:
    x = round(img_bgr.shape[1] - text_w - text_bg_pad*2 - text_bg_outline_width*2)
  elif x > 0 and x < 1:
    x = round(img_bgr.shape[1]*x - text_w/2 - text_bg_pad - text_bg_outline_width)
  # Add text to the image if desired.
  if not preview_only:
    # Draw a border around the background if desired.
    if text_bg_outline_color_bgr is not None:
      cv2.rectangle(img_bgr, (x,y), (x + text_w + 2*text_bg_outline_width + 2*text_bg_pad,
                                 y + text_h + 2*text_bg_outline_width + 2*text_bg_pad),
                    text_bg_outline_color_bgr, -1)
      x += text_bg_outline_width
      y += text_bg_outline_width
    
    # Draw the background shading.
    text_bg_color_bgr = text_bg_color_bgr or (100, 100, 100)
    cv2.rectangle(img_bgr, (x,y), (x + text_w + 2*text_bg_pad, y + text_h + 2*text_bg_pad),
                  text_bg_color_bgr, -1)
    x += text_bg_pad
    y += text_bg_pad
    
    # Draw the text.
    if text_color_bgr is None:
      text_color_bgr = (255, 255, 255)
    cv2.putText(img_bgr, text, (x, int(y + text_h + font_scale - 1)),
                font, font_scale, tuple(text_color_bgr), font_thickness)
  
  return (text_w, text_h, font_scale, (x, y))




###############################
# Segmentations Class
###############################

class Segmentations:
  ###############################
  # Initialization
  ###############################
  
  def __init__(self, h5_filepath=None,
               mask_shape=None, max_whale_index=None,
               video_filepaths=None, # a dictionary mapping a key to a filepath (example: {'myvid1': my_vid_1.mp4, 'myvid2': my_vid_2.mp4})
               num_video_frames_to_save=0, # -1 to save all frames
               video_fps=30, video_compression=17, video_preset='veryfast'):
    self._h5_filepath = h5_filepath
    self._max_whale_index = max_whale_index
    self._video_filepaths = video_filepaths if video_filepaths is not None else {}
    self._video_frame_dirs = {}
    self._num_video_frames_to_save = num_video_frames_to_save
    if self._num_video_frames_to_save != 0:
      for (video_key, video_filepath) in self._video_filepaths.items():
        (video_base_dir, video_filename) = os.path.split(video_filepath)
        self._video_frame_dirs[video_key] = os.path.join(video_base_dir, 'frame_images_%s' %
                                                           os.path.splitext(video_filename)[0])
        os.makedirs(self._video_frame_dirs[video_key], exist_ok=True)
        for image_filepath in glob.glob(os.path.join(self._video_frame_dirs[video_key], '*.jpg')):
          os.remove(image_filepath)
    
    self._video_fps = video_fps
    # Specify video compression.
    # Lossless may not be compatible with all players.
    #  If using lossless, preset should probably be ultrafast (faster encoding) or veryslow (better compression).
    #  For visually lossless but not technically lossless, recommend compression of around 17.
    # See https://trac.ffmpeg.org/wiki/Encode/H.264 for more information.
    self._video_compression = video_compression # range 0-51: 0 is lossless, default is 23
    self._video_preset = video_preset # faster easier to play back? [veryslow, slower, slow, medium, fast, veryfast, superfast, ultrafast]

    # Initialize state.
    self._dataset_expansion_size = 100
    h5_compression_level = 9
    self._bounding_box_keys = ['full', 'head', 'tail']
    self._datasets = {
      'masks': None,
      'centroids_xy': None,
      'orientations_rad_confidence': None,
    }
    self._bounding_box_key_to_name = lambda key: 'bounding_boxes_%s_4xy' % key
    self._bounding_box_names = [self._bounding_box_key_to_name(key) for key in self._bounding_box_keys]
    for bounding_box_name in self._bounding_box_names:
      self._datasets[bounding_box_name] = None
    self._num_frames = dict([(key, 0) for key in self._datasets])
    for video_key in self._video_filepaths:
      self._num_frames['video_%s' % video_key] = 0
    
    # Initialize the HDF5 output.
    self._h5_file = None
    if self._h5_filepath is not None:
      # Open the HDF5 file, creating it if it doesn't exist yet.
      self._h5_file = h5py.File(self._h5_filepath, 'a')
      # Point to existing datasets if this is an existing file,
      #  or create new ones if this is a new file.
      for dataset_key in self._datasets:
        if dataset_key in self._h5_file:
          self._datasets[dataset_key] = self._h5_file[dataset_key]
          self._num_frames[dataset_key] = self._datasets[dataset_key].shape[0]
          if dataset_key in ['centroids_xy', 'orientations_rad_confidence'] + self._bounding_box_names:
            self._max_whale_index = self._h5_file[dataset_key].shape[1]-1
        elif dataset_key == 'masks':
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     (self._dataset_expansion_size, *mask_shape),
                                                                     maxshape=(None, *mask_shape),
                                                                     dtype='uint8',
                                                                     chunks=True,
                                                                     compression='gzip',
                                                                     compression_opts=h5_compression_level) # 0-9, default is 4
        elif dataset_key in self._bounding_box_names:
          matrix_shape = [self._max_whale_index+1, 8]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     (self._dataset_expansion_size, *matrix_shape),
                                                                     maxshape=(None, *matrix_shape),
                                                                     dtype='int16',
                                                                     chunks=True,
                                                                     compression='gzip',
                                                                     compression_opts=h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'centroids_xy':
          matrix_shape = [self._max_whale_index+1, 2]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     (self._dataset_expansion_size, *matrix_shape),
                                                                     maxshape=(None, *matrix_shape),
                                                                     dtype='float',
                                                                     chunks=True,
                                                                     compression='gzip',
                                                                     compression_opts=h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'orientations_rad_confidence':
          matrix_shape = [self._max_whale_index+1, 2]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     (self._dataset_expansion_size, *matrix_shape),
                                                                     maxshape=(None, *matrix_shape),
                                                                     dtype='float',
                                                                     chunks=True,
                                                                     compression='gzip',
                                                                     compression_opts=h5_compression_level) # 0-9, default is 4
  
    # Open a reader for an existing video, or create an ffmpeg handle to write a new one.
    self._video_readers = {}
    self._ff_procs = {}
    for (video_key, video_filepath) in self._video_filepaths.items():
      if os.path.exists(video_filepath):
        self._video_readers[video_key] = decord.VideoReader(video_filepath)
        self._num_frames['video_%s' % video_key] = len(self._video_readers[video_key])
        self._ff_procs[video_key] = None
      else:
        self._ff_procs[video_key] = None # will be created when the first frame is provided
        self._video_readers[video_key] = None
    
  ###############################
  # General
  ###############################
  
  # Get the total number of frames.
  def get_num_frames(self):
    # num_frames = list(self._num_fames.values())
    # if np.any(np.diff(num_frames) != 0):
    #   raise AssertionError('Not all datasets in the file have the same number of frames.')
    # return num_frames[0]
    if self._h5_file is not None:
      return self._num_frames['masks']
    else:
      for (key, num_frames) in self._num_frames.items():
        if 'video' in key:
          return num_frames
    return None
  
  # Get the maximum number of frames.
  def get_num_whales(self):
    if self._max_whale_index is not None:
      return self._max_whale_index+1
    return None
  
  ###############################
  # Masks
  ###############################
  
  # Add a mask for the desired frame.
  def add_mask(self, frame_index, mask_matrix):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['masks']
    # Expand the dataset if needed.
    while len(dataset) < (frame_index+1):
      dataset.resize((len(dataset) + self._dataset_expansion_size, *dataset.shape[1:]))
    # Write the new entry.
    dataset[frame_index, :] = np.array(mask_matrix)
    self._num_frames['masks'] = max(frame_index + 1, self._num_frames['masks'])
    
  # Get a mask for a desired frame.
  def get_mask(self, frame_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['masks']
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return None
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    return np.squeeze(dataset[frame_index, :])
  
  # Get masks for all frames.
  # Will return an NxWxH matrix, where N is the number of frames, and WxH is the video resolution.
  def get_all_masks(self):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    return np.squeeze(self._datasets['masks'])
  
  ###############################
  # Bounding boxes
  ###############################
  
  # Add a bounding box for the desired frame and whale index.
  # bounding_box_4xy is 8 numbers: xy of each box corner in order base, leftUpper, top, rightUpper
  def add_bounding_box(self, bounding_box_key, frame_index, whale_index, bounding_box_4xy):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    dataset = self._datasets[dataset_name]
    # Expand the dataset if needed.
    while len(dataset) < (frame_index+1):
      dataset.resize((len(dataset) + self._dataset_expansion_size, *dataset.shape[1:]))
    # Write the new entry.
    dataset[frame_index, whale_index, :] = np.array(bounding_box_4xy)
    self._num_frames[dataset_name] = max(frame_index + 1, self._num_frames[dataset_name])
  
  # Get a bounding box for a desired frame and whale index.
  # bounding_boxes_4xy is 8 numbers: xy for each box corner
  # Will return None if there was no bounding box for the frame and whale index.
  def get_bounding_box_4xy(self, bounding_box_key, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    dataset = self._datasets[dataset_name]
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return None
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    bounding_boxes_4xy = np.squeeze(dataset[frame_index, whale_index, :])
    # Check if this was a dummy bounding box.
    if np.all(bounding_boxes_4xy == 0):
      return None
    return bounding_boxes_4xy
  
  # Get all bounding boxes for a desired frame.
  # Will return a dictionary mapping whale index to bounding box.
  # Each bounding box is 8 numbers: xy for each box corner
  # Values will be None if there was no bounding box for that whale index.
  def get_bounding_boxes_4xy(self, bounding_box_key, frame_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    bounding_boxes_4xy = OrderedDict()
    for whale_index in range(self._max_whale_index+1):
      bounding_boxes_4xy[whale_index] = self.get_bounding_box_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index)
    return bounding_boxes_4xy
  
  # Get all bounding boxes.
  # Will return an NxIx8 matrix, where N is the number of frames and I is the max whale index.
  # result[frame, whale, :] will be all 0 if there was no bounding box for that frame index and whale index.
  def get_all_bounding_boxes_4xy(self, bounding_box_key):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    return np.squeeze(self._datasets[dataset_name])
  
  ###############################
  # Centroids
  ###############################
  
  # Add a centroid of the mask for the desired frame and whale index.
  # centroid_yx is 2 numbers: (y, x)
  #  This can be the direct output of props.centroid if using skimage.measure.regionprops
  def add_centroid(self, frame_index, whale_index, centroid_yx):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['centroids_xy']
    # Expand the dataset if needed.
    while len(dataset) < (frame_index+1):
      dataset.resize((len(dataset) + self._dataset_expansion_size, *dataset.shape[1:]))
    # Write the new entry.
    dataset[frame_index, whale_index, :] = np.array(centroid_yx)[[1,0]]
    self._num_frames['centroids_xy'] = max(frame_index + 1, self._num_frames['centroids_xy'])
  
  # Get a centroid for a desired frame and whale index.
  # Will return None if there was no vector for the frame and whale index.
  def get_centroid_xy(self, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['centroids_xy']
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return None
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    centroid_xy = np.squeeze(dataset[frame_index, whale_index, :])
    # Check if this was a dummy vector.
    if np.all(centroid_xy == 0):
      return None
    return centroid_xy
  
  # Get all centroids for a desired frame.
  # Will return a dictionary mapping whale index to centroid.
  # Values will be None if there was no direction vector for that whale index.
  def get_centroids_xy(self, frame_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    centroids_xy = OrderedDict()
    for whale_index in range(self._max_whale_index+1):
      centroids_xy[whale_index] = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
    return centroids_xy
  
  # Get all centroids.
  # Will return an NxIx2 matrix, where N is the number of frames and I is the max whale index.
  # Each centroid is (x,y)
  # result[frame, whale, :] will be all 0 if there was no direction vector for that frame index and whale index.
  def get_all_centroids_xy(self):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    return np.squeeze(self._datasets['centroids_xy'])
  
  ###############################
  # Orientations
  ###############################
  
  # Add an orientation angle of the mask for the desired frame and whale index.
  def add_orientation(self, frame_index, whale_index, orientation_rad, orientation_confidence):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['orientations_rad_confidence']
    # Expand the dataset if needed.
    while len(dataset) < (frame_index+1):
      dataset.resize((len(dataset) + self._dataset_expansion_size, *dataset.shape[1:]))
    # Write the new entry.
    dataset[frame_index, whale_index, :] = np.squeeze(np.array([orientation_rad, orientation_confidence]))
    self._num_frames['orientations_rad_confidence'] = max(frame_index + 1, self._num_frames['orientations_rad_confidence'])
  
  # Get an orientation angle for a desired frame and whale index.
  # Will return (orientation_rad, orientation_confidence)
  # Will return (None, None) if there was no vector for the frame and whale index.
  def get_orientation_rad_confidence(self, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    dataset = self._datasets['orientations_rad_confidence']
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return (None, None)
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    (orientation_rad, orientation_confidence) = np.squeeze(dataset[frame_index, whale_index, :])
    # Check if this was a dummy vector.
    centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
    if centroid_xy is None:
      return (None, None)
    return (orientation_rad, orientation_confidence)
  
  # Get all orientations for a desired frame.
  # Will return a dictionary mapping whale index to (orientation_rad, orientation_confidence).
  # Values will be None if there was no direction vector for that whale index.
  def get_orientations_rad_confidence(self, frame_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    orientations_rad_confidence = OrderedDict()
    for whale_index in range(self._max_whale_index+1):
      orientations_rad_confidence[whale_index] = self.get_orientation_rad_confidence(frame_index=frame_index, whale_index=whale_index)
    return orientations_rad_confidence
  
  # Get all orientations.
  # Will return an NxIx2 matrix, where N is the number of frames, I is the max whale index,
  #   and 2 elements are (orientation_rad, orientation_confidence)
  # result[frame, whale, :] will be 0 if there was no segmentation for that frame index and whale index.
  #  So if an entry is 0, also do get_centroid_xy for that entry; if that is None, then the orientation is a dummy.
  def get_all_orientations_rad_confidence(self):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    return np.squeeze(self._datasets['orientations_rad_confidence'])
  
  ###############################
  # Segmented Images
  ###############################
  
  # Add a segmented image to the video.
  def add_segmented_image(self, video_key, frame_index, img, img_format='rgb', write_frame_index=True):
    # if self._video_readers[video_key] is not None:
    #   raise AssertionError('Cannot currently add to a video that existed at startup')
    if self._video_filepaths[video_key] is None:
      raise AssertionError('No video filepath was provided for key [%s]' % video_key)
    # Load the image if a filepath was provided.
    if isinstance(img, str):
      img = load_image(img)
      img = np.squeeze(img[:,:,0:3])
      img = np.ascontiguousarray(img)
      img_format = 'rgb'
    # Only use the first three image channels.
    img = np.squeeze(img[:,:,0:3])
    # Create the FFMPEG process if needed.
    (img_height, img_width, img_depth) = img.shape
    if self._ff_procs[video_key] is None:
      self._ff_procs[video_key] = (
            ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='%sx%s'%(int(img_width), int(img_height)),
                   r=self._video_fps, # assume a constant frame rate. NOTE: If put "r" as an output argument, ffmpeg will add/drop frames to achieve the target rate
                   )
            .output(self._video_filepaths[video_key],
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    crf=self._video_compression,
                    preset=self._video_preset,
                    # h264 needs dimensions divisible by 2,
                    # so add a filter that pads the bottom and right as needed
                    vf="pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
                    )
            .run_async(pipe_stdin=True)
        )
    # Add the frame number if desired.
    if write_frame_index:
      draw_text_on_image(img, 'Frame index: %6d' % frame_index,
                         pos=(-1, -1),
                         font_scale=None, text_width_ratio=0.2,
                         font_thickness=1, font=cv2.FONT_HERSHEY_DUPLEX,
                         text_color_bgr=(0, 0, 0),
                         text_bg_color_bgr=(200, 200, 200),
                         text_bg_outline_color_bgr=None,
                         text_bg_pad_width_ratio=0.03,
                         preview_only=False,
                         )
    # Convert the image if needed.
    if img_format.lower().strip() == 'bgr':
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Write black frames if needed to get to the target frame index.
    while self._num_frames['video_%s' % video_key] < frame_index:
      img_black = 0*img
      self._ff_procs[video_key].stdin.write(img_black.astype(np.uint8).tobytes())
      self._num_frames['video_%s' % video_key] += 1
    # Write the new frame to the video.
    self._ff_procs[video_key].stdin.write(img.astype(np.uint8).tobytes())
    self._num_frames['video_%s' % video_key] += 1
    
    # Save the frame as an image if desired.
    if video_key in self._video_frame_dirs:
      images_dir = self._video_frame_dirs[video_key]
      # Delete the oldest image if there are more than desired in the folder.
      images_saved = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
      if self._num_video_frames_to_save > 0 and len(images_saved) >= self._num_video_frames_to_save:
        os.remove(images_saved[0])
      # Save the new image.
      image_filepath = os.path.join(images_dir, '%s_frame_%06d.jpg' % (video_key, frame_index))
      cv2.imwrite(image_filepath, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
  # Get a mask for a desired frame.
  def get_segmented_image(self, video_key, frame_index):
    if self._ff_procs[video_key] is not None:
      raise AssertionError('Cannot currently read a video while it is being created')
    if self._video_filepaths[video_key] is None:
      raise AssertionError('No video filepath was provided for key [%s]' % video_key)
    if self._video_readers[video_key] is None:
      raise AssertionError('No video reader was opened for key [%s]' % video_key)
    if frame_index < 0 or frame_index >= self._num_frames['video_%s' % video_key]:
      raise ValueError('Invalid frame index %d for video [%s] with %d total frames' % (frame_index, video_key, self._num_frames['video_%s' % video_key]))
    return self._video_readers[video_key][frame_index].asnumpy()
  
  ###############################
  # Cleanup
  ###############################
  
  def quit(self):
    if self._h5_file is not None:
      # Resize the datasets to remove extra empty rows.
      for (dataset_key, dataset) in self._datasets.items():
        dataset.resize((self._num_frames[dataset_key], *dataset.shape[1:]))
      # Close the file
      self._h5_file.close()
      self._h5_file = None
    for ff_proc in self._ff_procs.values():
      if ff_proc is not None:
        ff_proc.stdin.close()
        ff_proc.wait()
    
  def close(self):
    self.quit()

  def __del__(self):
    self.close()
