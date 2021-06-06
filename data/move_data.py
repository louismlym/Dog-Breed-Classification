import shutil
import glob
import random
import os
import pandas as pd

IMAGE_LABEL = "./labels.csv"
TRAIN_PATH = "./train"
VAL_PATH = "./val"

def sample_images(percentage, old_path, new_path):
  images = glob.glob(old_path + '/*')
  num_images = len(images)
  sample_images = random.sample(images, k=int(num_images * percentage))
  for old_dir in sample_images:
    new_dir = new_path + '/' + os.path.basename(old_dir)
    shutil.move(old_dir, new_dir)

def make_image_folder_by_label(labels_path, images_path, name_col, label_col):
  csv = pd.read_csv(labels_path)
  img_to_label = {}
  for index, row in csv.iterrows():
    img_to_label[row[name_col]] = row[label_col]

  for file in os.listdir(images_path):
    old_dir = images_path + "/" + file
    img_name = os.path.splitext(file)[0]
    if img_name in img_to_label:
      label = img_to_label[img_name]
      new_dir = images_path + '/' + label + '/' + file
      os.makedirs(os.path.dirname(new_dir), exist_ok=True)
      shutil.move(old_dir, new_dir)

# sample_images(0.1, TRAIN_PATH, VAL_PATH)  # sample 10% of training images to be validation images
# make_image_folder_by_label(IMAGE_LABEL, VAL_PATH, 'id', 'breed')    # move each image into its own belonging label folder
# make_image_folder_by_label(IMAGE_LABEL, TRAIN_PATH, 'id', 'breed')  # move each image into its own belonging label folder