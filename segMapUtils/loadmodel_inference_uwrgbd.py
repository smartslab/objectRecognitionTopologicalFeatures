import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf


# %%
# =============================================================================
# Variables to set
# 
# tarballpath is the path to .tar.xz file containing the appropriate trained Deeplab model
# 
# img_dir is the path to the directory which contains images for which segmentation map 
# (either scene segmentation map or object segmentation map) are to be generated
# 
# 
# save_dir is the path to the directory where the generated segmentation maps are to be stored
# 
# =============================================================================

# %%

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def save_segmentation(image, seg_map, filename, savedir):
  """Visualizes input image, segmentation map and overlay view."""
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imsave(savedir+os.path.splitext(filename)[0] + '.png',seg_image)


from PIL import ImageEnhance

def factor_computation(img):
    if not(isinstance(img,np.ndarray)):   
        img = np.array(img)
    
    hist_img  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    histr = cv2.calcHist([hist_img],[0],None,[256],[0,256]) 
    freq = np.squeeze(np.array(histr))
    val = np.arange(0,256)
    avg = np.average(val, weights=freq)
    dev = freq * (val - avg) ** 2
    std = np.sqrt(dev.sum()/(freq.sum()-1))
    
    bright_factor = 128/avg
    contrast_factor = 128/3/std
    return bright_factor,contrast_factor


def run_inference(imagepath,filename,savedir):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(imagepath)

  except IOError:
    print('Cannot read image', filename)
    return

  print('running deeplab on image')
  resized_im, seg_map = MODEL.run(original_im)

  save_segmentation(resized_im, seg_map,filename,savedir)


tarballpath = './datasets/uwis/exp/train_on_trainval_set/export.tar.xz'
img_dir = './JPEGImages_livingroom/'
save_dir = './sceneSegMaps_livingroom/'


MODEL = DeepLabModel(tarballpath)
print('model loaded successfully!')



import fnmatch,os


image_files = fnmatch.filter(os.listdir(img_dir), '*.jpg')

for i in range(len(image_files)):
 	filename = image_files[i]
 	testimagepath = img_dir+image_files[i]
 	run_inference(testimagepath, filename,savedir)
