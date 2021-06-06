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

sample_images(0.1, TRAIN_PATH, VAL_PATH)

# labels = pd.read_csv(IMAGE_LABEL)
# iter = labels.iterrows()

# imgNameToLabel = {}
# for index, row in labels.iterrows():
#   imgNameToLabel[row['id']] = row['breed']

# count = 0

# for trainFile in os.listdir(TRAIN_PATH):
#   imgName = os.path.splitext(trainFile)[0]
#   if imgName in imgNameToLabel:
#     label = imgNameToLabel[imgName]
    
#     count = count + 1
#     print(imgNameToLabel[imgName])
#   if count >= 10:
#     break

# print(count)

