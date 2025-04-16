
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
# [can add additional updates and authors as desired]
#
############

import cv2
import decord
from PIL import Image

try:
  import ffmpeg
except:
  pass

import os
from datetime import datetime

import dateutil.parser
import numpy as np

try:
  import pyqtgraph
  from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
  from pyqtgraph.Qt.QtGui import QImage, QPixmap
  have_pyqt = True
except:
  have_pyqt = False
  
from collections import OrderedDict

############################################
# TIME
############################################

# Convert time as seconds since epoch to a human-readable string.
# timezone_offset_s is offset to add to time_s to convert from local time to UTC time.
# timezone_offset_str is the offset string in format HHMM, and may be negative.
# For example, for Eastern Daylight Time which is UTC-4:
#   timezone_offset_s = -14400
#   timezone_offset_str = '-0400'
def time_s_to_str(time_s, timezone_offset_s=None, timezone_offset_str=None,
                  use_current_local_time=True, use_current_utc_time=False,
                  date_str_format='%Y-%m-%d %H:%M:%S.%f',
                  date_str_include_timezone_offset=True):
  # Use the local timezone if none was provided.
  if timezone_offset_s is None:
    if use_current_utc_time:
      timezone_offset_s = 0
      timezone_offset_str = None
    elif use_current_local_time:
      is_dst = time.daylight and time.localtime().tm_isdst > 0
      timezone_offset_s = -(time.altzone if is_dst else time.timezone)
      timezone_offset_str = None
  if timezone_offset_str is None:
    timezone_offset_str = '%s%02d%02d' % ('-' if timezone_offset_s < 0 else '', int(abs(timezone_offset_s)/3600), int((abs(timezone_offset_s) % 3600)/60))
  # Get "UTC" time, which is actually local time because we will do the timezone offset first.
  time_datetime = datetime.utcfromtimestamp(time_s + timezone_offset_s)
  # Format the string then add the local offset string.
  return time_datetime.strftime('%s%s' % (date_str_format, ' ' + timezone_offset_str if date_str_include_timezone_offset else ''))

# Convert from a human-readable time string to time as seconds since epoch.
# The time string should include a timezone offset if applicable, for example '0400' for EDT.
def time_str_to_time_s(time_str):
  time_datetime = dateutil.parser.parse(time_str)
  return time_datetime.timestamp()

############################################
# DISTANCE
############################################

# Convert a GPS position to x/y distances from a reference location.
# longitude_deg and latitude_deg may be numpy arrays to convert multiple locations.
# The default reference location is the Mango House in Dominica.
# The default units for returned distances are meters.
# Returns (x, y) where x is longitude distance and y is latitude distance.
def gps_to_distance(longitude_deg, latitude_deg, reference_location_lonLat_deg=(-61.373179, 15.306914), units='m'):
  return (longitude_to_distance(longitude_deg, reference_location_lonLat_deg=reference_location_lonLat_deg, units=units),
          latitude_to_distance(latitude_deg, reference_location_lonLat_deg=reference_location_lonLat_deg, units=units))

# Convert a GPS longitude to meters from a reference location.
# longitude_deg may be a list or numpy array to convert multiple values.
# The default reference location is the Mango House in Dominica.
# The default units for returned distances are meters.
def longitude_to_distance(longitude_deg, reference_location_lonLat_deg=(-61.373179, 15.306914), units='m'):
  if isinstance(longitude_deg, (list, tuple)):
    longitude_deg = np.array(longitude_deg)
  conversion_factor_lon_to_km = (40075 * np.cos(np.radians(reference_location_lonLat_deg[1])) / 360) # From simplified Haversine formula: https://stackoverflow.com/a/39540339
  if units.lower() in ['km', 'kilometer', 'kilometers']:
    conversion_factor = conversion_factor_lon_to_km
  elif units.lower() in ['m', 'meter', 'meters']:
    conversion_factor = conversion_factor_lon_to_km*1000.0
  else:
    raise ValueError('Unknown units [%s]' % units)
  return (longitude_deg - reference_location_lonLat_deg[0])*conversion_factor

# Convert a GPS latitude to meters from a reference location.
# latitude_deg may be a list or numpy array to convert multiple values.
# The default reference location is the Mango House in Dominica.
# The default units for returned distances are meters.
def latitude_to_distance(latitude_deg, reference_location_lonLat_deg=(-61.373179, 15.306914), units='m'):
  if isinstance(latitude_deg, (list, tuple)):
    latitude_deg = np.array(latitude_deg)
  conversion_factor_lat_to_km = (111.32) # From simplified Haversine formula: https://stackoverflow.com/a/39540339
  if units.lower() in ['km', 'kilometer', 'kilometers']:
    conversion_factor = conversion_factor_lat_to_km
  elif units.lower() in ['m', 'meter', 'meters']:
    conversion_factor = conversion_factor_lat_to_km*1000.0
  else:
    raise ValueError('Unknown units [%s]' % units)
  return (latitude_deg - reference_location_lonLat_deg[1])*conversion_factor

############################################
# FILES AND TYPES
############################################

def get_file_extension(filepath):
  if not isinstance(filepath, str):
    return None
  file_extension = os.path.splitext(filepath)[-1]
  file_extension = file_extension.lower()
  return file_extension

def is_video(filepath_or_data):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, (cv2.VideoCapture, decord.VideoReader)):
    return True
  return get_file_extension(filepath_or_data) in ['.mp4', '.mov', '.avi', '.lrv', '.lrf'] or False

def is_image(filepath_or_data, enforce_dtype_uint8=True):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, np.ndarray):
    if filepath_or_data.ndim != 3:
      return False
    if enforce_dtype_uint8 and (filepath_or_data.dtype != np.uint8):
      return False
    return True
  return get_file_extension(filepath_or_data) in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] or False

def is_audio(filepath_or_data, max_num_channels=2):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, np.ndarray):
    return filepath_or_data.squeeze().ndim <= max_num_channels
  return get_file_extension(filepath_or_data) in ['.wav'] or False

def is_drone_data(filepath_or_data):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, dict) and 'altitude_relative_m' in filepath_or_data:
    return True
  return get_file_extension(filepath_or_data) in ['.srt'] or False

def is_coda_annotations(filepath_or_data):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, dict) and 'coda_start_times_s' in filepath_or_data:
    return True
  if isinstance(filepath_or_data, str):
    if 'coda_annotations' in filepath_or_data and get_file_extension(filepath_or_data) in ['.csv']:
      return True
  return False

def is_click_detections(filepath_or_data):
  if filepath_or_data is None:
    return False
  if isinstance(filepath_or_data, dict) and 'click_times_s' in filepath_or_data:
    return True
  if isinstance(filepath_or_data, str):
    if 'click_detections' in filepath_or_data and get_file_extension(filepath_or_data) in ['.csv']:
      return True
  return False
  
############################################
# IMAGES / VIDEOS
############################################

# Load an image from file.
# Optionally scale the image to a target size.
# Using the PIL method is fastest, since it can draft the downscaling during loading if it is a JPG.
# Will maintain the image's aspect ratio when scaling.
def load_image(filepath, target_width=None, target_height=None, method='pil'):
  img = None
  if method.lower() == 'opencv':
    img = cv2.imread(filepath)
    if target_width is not None or target_height is not None:
      img = scale_image(img, target_width=target_width, target_height=target_height, maintain_aspect_ratio=True)
  elif method.lower() == 'pil':
    img = Image.open(filepath)
    if target_width is not None and target_height is None:
      (img_width, img_height) = img.size
      target_height = int(target_width * img_height/img_width)
    if target_width is None and target_height is not None:
      (img_width, img_height) = img.size
      target_width = int(target_height * img_width/img_height)
    if target_width is not None and target_height is not None:
      img.draft('RGB', (int(target_width), int(target_height)))
      #print('target:', (target_width, target_height), '  drafted:', img.size, os.path.basename(filepath))
    img = np.asarray(img)
    if target_width is not None or target_height is not None:
      img = scale_image(img, target_width=target_width, target_height=target_height, maintain_aspect_ratio=True)
  return img

# Open a video file.
# Using decord is fastest and most accurate for frame seeking.
# target_width and target_height are only used for the 'decord' method.
#  Will then scale the frames as they are loaded to the target size (maintaining aspect ratio).
def get_video_reader(filepath, target_width=None, target_height=None, method='decord'):
  video_reader = None
  frame_rate = None
  num_frames = None
  if method.lower() == 'opencv':
    video_reader = cv2.VideoCapture(filepath)
    frame_rate = video_reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
  elif method.lower() == 'decord':
    video_reader = decord.VideoReader(filepath)
    if target_width is not None or target_height is not None:
      img = video_reader[0].asnumpy()
      img = scale_image(img, target_width, target_height)
      video_reader = decord.VideoReader(filepath, width=img.shape[1], height=img.shape[0])
    frame_rate = video_reader.get_avg_fps()
    num_frames = len(video_reader)
  return (video_reader, frame_rate, num_frames)
  
# Load a specified frame from a video reader.
# The video reader should be an OpenCV VideoCapture or Decord VideoReader object.
# Will return None if there was an issue fetching the frame.
def load_frame(video_reader, frame_index, target_width=None, target_height=None):
  img = None
  if isinstance(video_reader, cv2.VideoCapture):
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, img = video_reader.read()
    if success and (target_width is not None and target_height is not None):
      img = scale_image(img, target_width=target_width, target_height=target_height)
  elif isinstance(video_reader, decord.VideoReader):
    try:
      img = video_reader[frame_index].asnumpy()
      success = (img is not None)
    except:
      img = None
      success = False
    if success and (target_width is not None and target_height is not None):
      img = scale_image(img, target_width=target_width, target_height=target_height)
  return img
  
# Convert an OpenCV image to a PyQtGraph Pixmap.
def cv2_to_pixmap(cv_image):
  height, width, channel = cv_image.shape
  bytes_per_line = 3 * width
  q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
  return QPixmap.fromImage(q_image)

# Convert a PyQtGraph QImage to a numpy array.
def qimage_to_numpy(qimg):
  img = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB32)
  ptr = img.bits()
  ptr.setsize(img.sizeInBytes())
  arr = np.array(ptr).reshape(img.height(), img.width(), 4)  #  Copies the data
  return arr

# Scale an image to fit within a target width and height.
# If maintaining the image's aspect ratio (which is the default),
#   Will scale to the largest size that fits within the target size.
# If the image size already meets the target criteria,
#   will return the original image (this call will not incur any delays beyond the size check).
# The input can be a numpy array or a PyQtGraph QPixmap object.
# Will also measure the total time spent in this method, for profiling purposes.
import time

duration_s_scaleImage = 0
def get_duration_s_scaleImage():
  global duration_s_scaleImage
  return duration_s_scaleImage
def scale_image(img, target_width=None, target_height=None, maintain_aspect_ratio=True):
  global duration_s_scaleImage
  t0 = time.time()
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
      duration_s_scaleImage += time.time() - t0
      return res
    else:
      # If not maintaining the aspect ratio, scale each dimension by its computed factor.
      # If one of the dimensions was not provided, assume it should be kept at the original size.
      if scale_factor_byWidth is None:
        scale_factor_byWidth = 1
      if scale_factor_byHeight is None:
        scale_factor_byHeight = 1
      if scale_factor_byWidth != 1 or scale_factor_byHeight != 1:
        res = cv2.resize(src=img, dsize=(0,0), fx=scale_factor_byWidth, fy=scale_factor_byHeight)
      else:
        # Do nothing if the image size is already as desired.
        res = img
      duration_s_scaleImage += time.time() - t0
      return res
  # Scale a QPixmap object.
  elif isinstance(img, QPixmap):
    if maintain_aspect_ratio:
      res = img.scaled(target_width, target_height,
                       aspectRatioMode=pyqtgraph.QtCore.Qt.AspectRatioMode.KeepAspectRatio)
      duration_s_scaleImage += time.time() - t0
      return res
    else:
      res = img.scaled(target_width, target_height,
                       aspectRatioMode=pyqtgraph.QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
      duration_s_scaleImage += time.time() - t0
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
import time

duration_s_drawText = 0
def get_duration_s_drawText():
  global duration_s_drawText
  return duration_s_drawText
def draw_text_on_image(img_bgr, text, pos=(0, 0),
                       font_scale=8, text_width_ratio=None, text_height_ratio=None,
                       font_thickness=1, font=cv2.FONT_HERSHEY_DUPLEX,
                       text_color_bgr=None,
                       text_bg_color_bgr=None, text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0.03,
                       preview_only=False,
                       ):
  global duration_s_drawText
  t0 = time.time()
  # If desired, compute a font scale based on the target width ratio.
  font_scale_byWidthRatio = None
  font_scale_byHeightRatio = None
  if font_scale is None and text_width_ratio is not None:
    if len(text) > 0:
      target_text_w = text_width_ratio * img_bgr.shape[1]
      font_scale_byWidthRatio = 0
      text_w = 0
      while text_w < target_text_w:
        font_scale_byWidthRatio += 0.2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale_byWidthRatio, font_thickness)
      font_scale_byWidthRatio -= 0.2
    else:
      font_scale_byWidthRatio = 1
  # If desired, compute a font scale based on the target height ratio.
  if font_scale is None and text_height_ratio is not None:
    if len(text) > 0:
      target_text_h = text_height_ratio * img_bgr.shape[0]
      font_scale_byHeightRatio = 0
      text_h = 0
      while text_h < target_text_h:
        font_scale_byHeightRatio += 0.2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale_byHeightRatio, font_thickness)
      font_scale_byHeightRatio -= 0.2
    else:
      font_scale_byHeightRatio = 1
  # If both ratios were provided, use the smaller font.
  if font_scale is None:
    if font_scale_byHeightRatio is not None and font_scale_byWidthRatio is not None:
      font_scale = min(font_scale_byHeightRatio, font_scale_byWidthRatio)
    elif font_scale_byHeightRatio is not None:
      font_scale = font_scale_byHeightRatio
    elif font_scale_byWidthRatio is not None:
      font_scale = font_scale_byWidthRatio
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
  
  duration_s_drawText += time.time() - t0
  return (text_w, text_h, font_scale, (x, y))

# Rotate an image by a given angle.
# Source: https://stackoverflow.com/a/47248339
def rotate_image(img, angle_rad):
  size_reverse = np.array(img.shape[1::-1]) # swap x with y
  M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), np.degrees(angle_rad), 1.)
  MM = np.absolute(M[:,:2])
  size_new = MM @ size_reverse
  M[:,-1] += (size_new - size_reverse) / 2.
  return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

# Compress a video to the target bitrate.
# The target bitrate in bits per second will include both video and audio.
def compress_video(input_filepath, output_filepath, target_total_bitrate_b_s):
  # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
  min_audio_bitrate = 32000
  max_audio_bitrate = 256000
  
  # Open a probe to the input video.
  probe = ffmpeg.probe(input_filepath)
  
  # Check if ausio will be included.
  audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
  if len(audio_streams) > 0:
    audio_bitrate = sum(float(audio_stream['bit_rate']) for audio_stream in audio_streams)
    
    if 10 * audio_bitrate > target_total_bitrate_b_s:
      audio_bitrate = target_total_bitrate_b_s / 10
    if audio_bitrate < min_audio_bitrate < target_total_bitrate_b_s:
      audio_bitrate = min_audio_bitrate
    elif audio_bitrate > max_audio_bitrate:
      audio_bitrate = max_audio_bitrate
    
    video_bitrate = target_total_bitrate_b_s - audio_bitrate
  else:
    audio_bitrate = None
    video_bitrate = target_total_bitrate_b_s
  
  # Compress!
  i = ffmpeg.input(input_filepath)
  # Pass 1
  ffmpeg_args = {
    'c:v': 'libx264',
    'b:v': video_bitrate,
    'pass': 1,
    'f': 'mp4',
    'loglevel':'error',
  }
  ffmpeg.output(i, os.devnull,
                **ffmpeg_args
                ).overwrite_output().run()
  # Pass 2
  ffmpeg_args = {
    'c:v': 'libx264',
    'b:v': video_bitrate,
    'pass': 2,
    'c:a': 'aac',
    'loglevel': 'error',
  }
  if len(audio_streams) > 0:
    ffmpeg_args['b:a'] = audio_bitrate
  ffmpeg.output(i, output_filepath,
                **ffmpeg_args
                ).overwrite_output().run()

############################################
# Timeseries
############################################

# Compute a moving average of the given data.
# Centering can be 'leading', 'trailing', or 'centered'.
# Mode can be 'same' or 'valid'
#   If 'same', the output will be the same length as the input
#     but the initial and final inputs may be averaged over fewer elements than the full window.
#   If 'valid', will only return elements where the window fully fits on the signal.
# Returns (time_s, averaged_x)
def moving_average(time_s, x, window_duration_s, centering, mode='same'):
  # Compute the window size.
  signal_length = len(time_s)
  Fs = 1/np.mean(np.diff(time_s))
  window_length = min(len(x), round(window_duration_s*Fs))
  window = np.ones((window_length,))
  # Compute the full convolution.
  num_elements = np.convolve(np.ones_like(x), window, 'full')
  moving_average = np.convolve(x, window, 'full') / num_elements
  full_length = moving_average.shape[0]
  # Extract the appropriate subset and time vector.
  if mode.lower() == 'valid' and centering == 'leading':
    return (time_s[0:(signal_length - window_length + 1)],
            moving_average[(window_length-1):(full_length - window_length + 1)])
  if mode.lower() == 'valid' and centering == 'trailing':
    return (time_s[(window_length-1):],
            moving_average[(window_length-1):(full_length - window_length + 1)])
  if mode.lower() == 'valid' and centering == 'centered':
    start_index = round((window_length-1)/2)
    end_index = start_index + (full_length - 2*window_length + 1)
    return (time_s[start_index:(end_index+1)],
            moving_average[(window_length-1):(full_length - window_length + 1)])
  if mode.lower() == 'same' and centering == 'leading':
    return (time_s[:],
            moving_average[(window_length-1):])
  if mode.lower() == 'same' and centering == 'trailing':
    return (time_s[:],
            moving_average[0:(full_length - window_length + 1)])
  if mode.lower() == 'same' and centering == 'centered':
    start_index = round((window_length-1)/2)
    end_index = start_index + signal_length - 1
    return (time_s[:],
            moving_average[start_index:(end_index+1)])

# Find rising edges in the given signal.
# Will return the indexes of each rising edge,
#  where the data at the returned index is the first high point of the step.
def rising_edges(x, threshold=0.5, include_first_step_if_high=False, include_last_step_if_low=False):
  if not isinstance(x, np.ndarray):
    x = np.array(x)
  # Copied from https://stackoverflow.com/a/50365462
  edges = list(np.flatnonzero((x[:-1] < threshold) & (x[1:] > threshold))+1)
  # Explicitly handle edge cases.
  if x[0] > threshold and include_first_step_if_high:
    edges = [0] + edges
  if x[-1] < threshold and include_last_step_if_low:
    edges = edges + [len(x)-1]
  return np.array(edges)

# Find falling edges in the given signal.
# Will return the indexes of each falling edge,
#  where the data at the returned index is the first low point of the step.
def falling_edges(x, threshold=0.5, include_first_step_if_low=False, include_last_step_if_high=False):
  if not isinstance(x, np.ndarray):
    x = np.array(x)
  # Adapted from https://stackoverflow.com/a/50365462
  edges = list(np.flatnonzero((x[:-1] > threshold) & (x[1:] < threshold))+1)
  # Explicitly handle edge cases.
  if x[0] < threshold and include_first_step_if_low:
    edges = [0] + edges
  if x[-1] > threshold and include_last_step_if_high:
    edges = edges + [len(x)-1]
  return np.array(edges)

# Find nan entries and fill them with the previous non-nan value.
# Will allow leading nan entries to remain.
def fill_nans(x):
  nan_indexes_original = np.where(np.isnan(x))[0]
  nan_indexes = nan_indexes_original.copy()
  if np.isnan(x[0]):
    indexes_toAllow = []
    for i in range(len(x)):
      if np.isnan(x[i]):
        indexes_toAllow.append(i)
      else:
        break
    nan_indexes = np.delete(nan_indexes, indexes_toAllow)
  while nan_indexes.size > 0:
    x[nan_indexes] = x[nan_indexes-1]
    nan_indexes = np.where(np.isnan(x))[0]
    if np.isnan(x[0]):
      indexes_toAllow = []
      for i in range(len(x)):
        if np.isnan(x[i]):
          indexes_toAllow.append(i)
        else:
          break
      nan_indexes = np.delete(nan_indexes, indexes_toAllow)
  return nan_indexes_original

############################################
# Printing and Formatting Variables
############################################

# Cast all values in a (possibly nested) dictionary to strings.
# Will remove key-value pairs for values that cannot be easily converted to a string.
# If preserve_nested_dicts is True, will preserve nested structure but recursively convert their values.
#   Otherwise, will simply stringify the whole nested dictionary.
def convert_dict_values_to_str(d, preserve_nested_dicts=True):
  # Create a new dictionary that will be populated
  if isinstance(d, OrderedDict):
    d_converted = OrderedDict()
  else:
    d_converted = {}
  
  for (key, value) in d.items():
    # Recurse if the value is a dictionary
    if isinstance(value, dict) and preserve_nested_dicts:
      d_converted[key] = convert_dict_values_to_str(value, preserve_nested_dicts=preserve_nested_dicts)
    else:
      # Add the item to the new dictionary if its value is convertible to a string
      try:
        d_converted[key] = str(value)
      except:
        pass
  return d_converted

# Print a dictionary (recursively as appropriate).
def print_dict(d, level=0):
  print(get_dict_str(d, level=level))

# Get a string to display a dictionary (recursively as appropriate).
def get_dict_str(d, level=0):
  indent_root = ' '*level
  indent_keys =  indent_root + ' '
  msg = '%s{\n' % indent_root
  for (key, value) in d.items():
    msg += '%s %s: ' % (indent_keys, key)
    if isinstance(value, dict):
      msg += '\n'
      msg += get_dict_str(value, level+2) # one level for the key indent, one for advancing the level
    else:
      msg += '%s\n' % str(value)
  msg += '%s}\n' % indent_root
  return msg

# Print a variable and its type.
def print_var(var, name=None):
  print(get_var_str(var, name=name))

# Get a string to display a variable and its type.
def get_var_str(var, name=None):
  msg = ''
  if name is not None:
    msg += 'Variable "%s" of ' % name
  msg += 'Type %s: ' % type(var)
  msg += ''
  processed_var = False
  # Dictionary
  if isinstance(var, dict):
    msg += get_dict_str(var, level=3)
    processed_var = True
  # String
  if isinstance(var, str):
    msg += '"%s"' % var
    processed_var = True
  # Numpy array, if numpy has been imported
  try:
    if isinstance(var, np.ndarray):
      msg += '\n shape: %s' % str(var.shape)
      msg += '\n data type: %s' % str(var.dtype)
      msg += '\n %s' % str(var)
      processed_var = True
  except NameError:
    pass
  # Lists and tuples
  if isinstance(var, (list, tuple)):
    contains_non_numbers = False in [isinstance(x, (int, float)) for x in var]
    if contains_non_numbers:
      msg += '['
      for (i, x) in enumerate(var):
        msg += '\n %d: %s' % (i, get_var_str(x))
      msg += '\n ]'
    else:
      msg += '%s' % str(var)
    processed_var = True
  # Everything else
  if not processed_var:
    msg += '%s' % str(var)
    processed_var = True
  # Done!
  return msg.strip()

############################################
# Math
############################################

# Get the next multiple of a number above a specified target.
# For example, the next multiple of 5 above 23 would be 25.
def next_multiple(value, multiple_of):
  if int(value/multiple_of) == value/multiple_of:
    return value
  return (np.floor(value/multiple_of) + 1)*multiple_of

# Get the previous multiple of a number below a specified target.
# For example, the previous multiple of 5 below 23 would be 20.
def previous_multiple(value, multiple_of):
  if int(value/multiple_of) == value/multiple_of:
    return value
  return np.floor(value/multiple_of)*multiple_of