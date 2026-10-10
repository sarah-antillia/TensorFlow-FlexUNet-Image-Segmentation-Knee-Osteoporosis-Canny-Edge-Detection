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
# 2026/10/10 ImageMaskDatasetGenerator.py


import sys
import os
import cv2
import numpy as np
import glob
import traceback
import shutil

class ImageMaskDatasetGenerator:
  
  def __init__(self, resize=512, num_classes=3):
      self.RESIZE = (resize, resize)

      # GaussianBlur ksize
      self.GaussianBlur_ksize = (5, 5)
      self.CANNY_Threshold1 = 30
      self.CANNY_Threshold2 = 80
      self.BLUR_ksize       = (5,5)

      self.GRADES = ["Normal", "Osteopenia", "Osteoporosis",]
      if num_classes == 2:
        self.GRADES = [ "Osteopenia", "Osteoporosis",]
          
      # Default class_colors: num_classes=3
  
      # Default: class_color_mapping table for two classes
      self.MASK_RGB_COLORS = {"Normal":(0,255,0), "Osteopenia":(255,255,0), "Osteoporosis":(180,20,20),}
      if num_classes == 2:
        self.MASK_RGB_COLORS = {"Osteopenia":(0,255,0), "Osteoporosis":(255,255,0),}

  def generate(self, data_dir, output_images_dir, output_masks_dir):
     num   = len(self.GRADES)
     for i in range(num):
        grade      = self.GRADES[i]
        images_dir = os.path.join(data_dir, grade)
        if os.path.exists(images_dir):
          mask_color = self.MASK_RGB_COLORS[grade]
          self.generate_one(images_dir, grade, mask_color, output_images_dir, output_masks_dir)
       
  def generate_one(self, images_dir, grade, color, output_images_dir, output_masks_dir):
    image_files = sorted(glob.glob(images_dir + "/*.png"))
    for image_file in image_files:
      img = cv2.imread(image_file)
      img = cv2.resize(img, self.RESIZE)
   
      # Generate a canny edge mask
      canny_edge = self.canny_edge(img)

      # Apply the OTSU binarizer to the canny_edge.
      _, bin_img = cv2.threshold(canny_edge, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      inverted = 255 - bin_img
      
      if inverted.any() >0:
        basename = os.path.basename(image_file)
        filename = basename
        out_image_filepath = os.path.join(output_images_dir, filename)
        cv2.imwrite(out_image_filepath, img)
        print("Saved ", out_image_filepath)

        # Generate a colorized mask from the inverted mask
        colorized = self.colorize_mask(inverted, color)
        out_mask_filepath = os.path.join(output_masks_dir, filename)
        
        cv2.imwrite(out_mask_filepath, colorized)
        print("Saved ", out_mask_filepath)

        self.horizontal_flip(img, filename, output_images_dir)
        self.horizontal_flip(colorized, filename, output_masks_dir)

        #self.vertical_flip(img, filename, output_images_dir)
        #self.vertical_flip(colorized, filename, output_masks_dir)
           
      else:
         print("Skipped an empty mask!---")
         
  def colorize_mask(self, mask, rgb_color):
      h, w  = mask.shape[:2]
      (r, g, b) = rgb_color
      colorized = np.zeros((h, w, 3), dtype=np.uint8)
      colorized [np.equal(mask, 255)] = (b, g, r)
      return colorized
  
  def canny_edge(self, img):     
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                          
    sigma = 0                                                             
    img = cv2.GaussianBlur(img, self.GaussianBlur_ksize, sigma) 
    
    canny = cv2.Canny(img, threshold1=self.CANNY_Threshold1, threshold2=self.CANNY_Threshold2)
    canny = cv2.blur(canny, self.BLUR_ksize)
    return canny

  def horizontal_flip(self, image, filename, output_dir): 
    print("Horizontal_flip: shape image {}".format(image.shape))
    flipped = image
    if len(image.shape)==3:
      flipped =  image[:, ::-1, :]
    else:
      flipped =  image[:, ::-1, ]
    flipped_filename = "hflipped_" + filename
    filepath = os.path.join(output_dir, flipped_filename)   
    cv2.imwrite(filepath, flipped)
    print("Saved ", filepath)

  def vertical_flip(self, image, filename, output_dir):
    print("Vertical_flip: shape image {}".format(image.shape))
    flipped = image

    if len(image.shape) == 3:
      flipped = image[::-1, :, :]
    else:
      flipped = image[::-1, :, ]
    flipped_filename = "vflipped_" + filename
    filepath = os.path.join(output_dir, flipped_filename)   
    cv2.imwrite(filepath, flipped)
    print("Saved ", filepath)


if __name__ == "__main__":
 
  try:
    data_dir   = "./Knee_Osteoporosis_Cropped/"
    
    output_dir = "./Knee-Osteoporosis-Canny-Edge-Detection"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)
 
    generator = ImageMaskDatasetGenerator(resize = 512, num_classes = 3)
    generator.generate(data_dir, output_images_dir, output_masks_dir)
     
  except Exception as e:
    traceback.print_exc()

