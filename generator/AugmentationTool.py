# Copyright 2026 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDI66TIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2026/10/10 AugmentationTool.py

import os
import sys
import io
import cv2
import shutil
import glob
import SimpleITK as sitk

import numpy as np
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import traceback

class AugmentationTool:

  def __init__(self, resize=512, augmentation=True):
    self.RESIZE    = (resize, resize)
 
    self.seed     = 137
    self.W        = resize
    self.H        = resize
     
    self.file_format= ".png"
    self.augmentation = augmentation
    if self.augmentation:
      self.hflip    = False
      self.vflip    = False
      self.rotation = False
      self.ANGLES   = [90, 180, 270]

      self.deformation=True
      self.alpha    = 1300
      self.sigmoids = [7, ]
          
      self.distortion=True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5
      self.distortions           = [0.01, ]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)
      

  def generate(self, images_dir, inferred_masks_dir, output_images_dir, output_masks_dir):
    image_files = sorted(glob.glob(images_dir + "/*.png"))
    mask_files  = sorted(glob.glob(inferred_masks_dir + "/*.png"))
    n = len(image_files)
    m = len(mask_files)
    if n != m:
      raise Exception("Unmatched the number of images and masks")

    for i in range(m):
      
      mask_file  = mask_files[i]
      image_file = image_files[i]
      
      image    = cv2.imread(image_file)
      mask     = cv2.imread(mask_file)
      filename = os.path.basename(image_file)
      output_image_filepath = os.path.join(output_images_dir, filename)
      output_mask_filepath  = os.path.join(output_masks_dir, filename)
      cv2.imwrite(output_image_filepath, image)
      cv2.imwrite(output_mask_filepath,  mask)
      print("Saved ", output_image_filepath)
      print("Saved ", output_mask_filepath)

      if self.augmentation:
          self.augment(image, filename, output_images_dir, border=(0, 0, 0), mask=False)
          self.augment(mask, filename, output_masks_dir,  border=(0, 0, 0), mask=True)

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    border = image[2][2].tolist()
  
    print("---- border {}".format(border))
    if basename.startswith("Osteoporosis") or basename.startswith("Osteopenia"):
      print("Osteoporosis  or Osteopenia")
      #if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

      #if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))


    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))

  def deform(self, image, basename, output_dir): 
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    print("--- shape {}".format(shape))

    for sigmoid in self.sigmoids:
      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

      deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed_image = deformed_image.reshape(image.shape)

      image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, deformed_image)
      print("=== Saved deformed image file {}".format(image_filepath))
     
  # This method is based on the code of the following stackoverflow.com webstie:
  # https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/78031420#78031420
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file {}".format(output_file))



if __name__ == "__main__":
  try:
    images_dir = "./Curated/images/"
    masks_dir  = "./Curated/masks"

    output_dir = "./Knee-Osteoporosis-Canny-Edge-master"
    if os.path.exists(output_dir):
       shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    generator = AugmentationTool(resize=512, augmentation=True)
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()

   