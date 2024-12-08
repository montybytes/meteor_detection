"""
This script creates a training directory for the YOLOv8 multi-class detector
model and balances the dataset out to try and avoid overfitting on a single
class due to class imbalance.
"""

import os
import numpy as np
import random
import shutil

root = "dataset"

img_folder = "images"
label_folder = "labels"


files = os.listdir(os.path.join(root, label_folder))

random.shuffle(files)

broadbands = []
complex_meteor = []
simple_meteor = []


for txt_file in files:
    path = os.path.join(root, label_folder, txt_file)
    name, ext = os.path.splitext(path)

    # check if text file
    if ext != ".txt":
        continue

    # open text file
    with open(path) as f:
        lines = f.readlines()

        # check if 2 exists in class column
        for line in lines:
            cols = line.split(" ")
            if cols[0] == "2":
                if txt_file not in broadbands:
                    broadbands.append(txt_file)


for txt_file in files:
    path = os.path.join(root, label_folder, txt_file)
    name, ext = os.path.splitext(path)

    # check if text file
    if ext != ".txt":
        continue

    # open text file
    with open(path) as f:
        lines = f.readlines()

        # check if 1 exists in class column
        for line in lines:
            cols = line.split(" ")

            if cols[0] == "1":
                if (
                    txt_file not in complex_meteor
                    and txt_file not in broadbands
                    and len(complex_meteor) < len(broadbands)
                ):
                    complex_meteor.append(txt_file)


for txt_file in files:
    path = os.path.join(root, label_folder, txt_file)
    name, ext = os.path.splitext(path)

    # check if text file
    if ext != ".txt":
        continue

    # open text file
    with open(path) as f:
        lines = f.readlines()

        # check if 0 exists in class column
        for line in lines:
            cols = line.split(" ")

            if cols[0] == "0":
                if (
                    txt_file not in simple_meteor
                    and txt_file not in broadbands
                    and txt_file not in complex_meteor
                    and len(simple_meteor) < len(broadbands)
                ):
                    simple_meteor.append(txt_file)

split_index = int(len(broadbands) * 0.8)
print(split_index)

train_broadband, val_broadband = broadbands[:split_index], broadbands[split_index:]
train_complex, val_complex = complex_meteor[:split_index], complex_meteor[split_index:]
train_simple, val_simple = simple_meteor[:split_index], simple_meteor[split_index:]

def mover(array, subset):
    for label in array:
        name, ext = os.path.splitext(label)

        img_src = os.path.join(root, img_folder, "{}.png".format(name))
        img_dst = os.path.join("dataset_balanced", "images", subset, "{}.png".format(name))

        label_src = os.path.join(root, label_folder, label)
        label_dst = os.path.join("dataset_balanced", "labels", subset, label)

        shutil.copyfile(img_src, img_dst)
        shutil.copyfile(label_src, label_dst)
    return


mover(train_broadband, "train")
mover(val_broadband, "val")
mover(train_complex, "train")
mover(val_complex, "val")
mover(train_simple, "train")
mover(val_simple, "val")
