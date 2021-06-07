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

def get_top(num):
  counts = {}
  for file in os.listdir(TRAIN_PATH):
    counts[file] = len(os.listdir(TRAIN_PATH + '/' + file))

  for file in os.listdir(VAL_PATH):
    counts[file] += len(os.listdir(VAL_PATH + '/' + file))

  top = []
  for i in sorted(counts.items(), key = lambda kv:(-kv[1], kv[0]))[:num]:
    top.append(i[0])
  return top

def create_small_dataset(top, images_path, suffix):
  for folder in os.listdir(images_path):
    if folder not in top:
      continue
    for file in os.listdir(images_path + '/' + folder):
      old_path = images_path + '/' + folder + '/' + file
      new_path = images_path + suffix + '/' + folder + '/' + file
      os.makedirs(os.path.dirname(new_path), exist_ok=True)
      shutil.copy(old_path, new_path)

# sample_images(0.1, TRAIN_PATH, VAL_PATH)  # sample 10% of training images to be validation images
# make_image_folder_by_label(IMAGE_LABEL, VAL_PATH, 'id', 'breed')    # move each image into its own belonging label folder
# make_image_folder_by_label(IMAGE_LABEL, TRAIN_PATH, 'id', 'breed')  # move each image into its own belonging label folder
top10 = get_top(10)
create_small_dataset(top10, TRAIN_PATH, '_top10')
create_small_dataset(top10, VAL_PATH, '_top10')