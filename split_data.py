from typing import List, Any

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.utils import load_img
import shutil

data_file = './data/'
img_folder = './data/MAMe_data_256/data_256/'
data_folder = 'data/MAMe_metadata/'
data_csv = data_folder + 'MAMe_dataset.csv'
label_csv = data_folder + 'MAMe_labels.csv'
data_df = pd.read_csv(data_csv)
subsets = data_df['Subset']. unique().tolist()

for subset in subsets:
    subset_df = data_df[data_df['Subset'] == subset]
    print('{} {} examples in MAME dataset'.format(len(subset_df), subset))

classes = sorted(data_df['Medium'].unique())
num_classes = len(classes)

cls_folder_dict = {'train': dict(), 'val': dict(), 'test': dict()}
for subset in subsets:
    subset_folder = data_file + subset
    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)
    for cls in classes:
        cls_folder = subset_folder + '/' + cls + '/'
        cls_folder_dict[subset][cls] = cls_folder
        if not os.path.exists(cls_folder):
            os.mkdir(cls_folder)

cols = data_df.columns
image_files = data_df[cols[0]].values
image_subsets = data_df[cols[4]].values
image_cls = data_df[cols[1]].values

new_img_paths: List[Any] = []

for path, subset, cls in zip(image_files, image_subsets, image_cls):
    org_img_path = img_folder + path
    img_path = cls_folder_dict[subset][cls] + path
    new_img_paths.append(img_path)
    shutil.copyfile(org_img_path, img_path)

print('length of new images: ' + len(new_img_paths))
data_df['Organized_img_path'] = new_img_paths

data_df.head(5)
data_df.to_csv('Organized_MAMe_dataset.csv')









