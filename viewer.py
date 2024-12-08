"""
This Python script displays bounding boxes on a set of images to visualize detections
in relation to the ground truths.
"""

import cv2
import argparse
import pandas as pd

from os import listdir
from os.path import isfile, join
from sample import Sample

import lib.vision as vision

# Titles and labels used for the OpenCV window and trackbar
title = "Simple Meteor Detection Algorithm: Viewer"
img_index = "Image Index:"

# Global variables to hold image names, ground truths, and detections
img_names = None
truths = None
detections = None

# Setting up argument parsing for the script
parser = argparse.ArgumentParser()

parser.add_argument(
    "dir",
    default=None,
    help="The folder to display detections on",
)
parser.add_argument(
    "-l",
    "--labels",
    default=None,
    help="The labels file",
)
parser.add_argument(
    "-d",
    "--detections",
    default=None,
    help="The estimated detections file",
)

# Parse the provided command line arguments
args = parser.parse_args()


def get_path(img_name):
    """Function to get the full path of an image file"""

    if isfile(join(args.dir, img_name)):
        return join(args.dir, img_name)

    return None


def load_image(img_name):
    """Function to load an image using OpenCV"""
    path = get_path(img_name)
    img = cv2.imread(path)

    # If the image could not be loaded, display an error and exit
    if img is None:
        print("Error: Could not open or find the image: ", path)
        exit(0)

    return img


def load_file(path):
    """Function to load bounding box data from a CSV file and return list of
    sample objects"""
    df = pd.read_csv(path)

    # convert the dataframe to label the columns with required variables x2 and y2
    if "bbox_width" in df.columns or "bbox_height" in df.columns:
        df.rename({"bbox_x": "x1", "bbox_y": "y1"}, axis=1, inplace=True)
        df["x2"] = df["x1"] + df["bbox_width"]
        df["y2"] = df["y1"] + df["bbox_height"]

    # Keep only relevant columns
    if "image_name" in df.columns:
        df = df[["image_name", "x1", "y1", "x2", "y2"]].dropna()

    # Convert dataframe rows to a list of Sample objects
    return [Sample(s[0], s[1], s[2], s[3], s[4]) for s in df.to_numpy()]


def load_bboxes(img_name, sample_array):
    """Function to load bounding boxes for a specific image"""

    return [sample.bbox for sample in sample_array if img_name == sample.image_name]


def update_window(val=0):
    """Function to update the displayed window with the image and bounding boxes"""

    # Load the current image based on the trackbar value
    img = load_image(img_names[val])

    # Load the ground truth and detection bounding boxes for the current image
    estimated_bboxes = load_bboxes(img_names[val], detections)
    truth_bboxes = load_bboxes(img_names[val], truths)

    # Draw the bounding boxes: red for ground truth, green for detection estimates
    for box in truth_bboxes:
        vision.draw_bbox(img, box, color=(0, 0, 255), line_width=2)

    for box in estimated_bboxes:
        vision.draw_bbox(img, box, color=(0, 255, 0))

    cv2.imshow(title, img)

    return


def main():
    global img_names, truths, detections

    # Get the list of images in the specified directory
    img_names = listdir(args.dir)
    # Load the ground truth and predicted bounding boxes
    truths = load_file(args.labels)
    detections = load_file(args.detections)

    if len(img_names) == 0:
        print("No files found in given directory")
        exit(0)

    # Set up an OpenCV window and trackbar for navigating through images
    cv2.namedWindow(title)
    cv2.createTrackbar(img_index, title, 0, len(img_names) - 1, update_window)
    cv2.waitKey()

    return


if __name__ == "__main__":
    main()
