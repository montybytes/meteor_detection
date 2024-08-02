import cv2
import argparse
import pandas as pd

from os import listdir
from os.path import isfile, join
from sample import Sample

import lib.vision as vision


title = "Simple Meteor Detection Algorithm: Viewer"
img_index = "Image Index:"

img_names = None
truths = None
detections = None

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

args = parser.parse_args()


def get_path(img_name):
    if isfile(join(args.dir, img_name)):
        return join(args.dir, img_name)

    return None


def load_image(img_name):
    path = get_path(img_name)
    img = cv2.imread(path)

    if img is None:
        print("Error: Could not open or find the image: ", path)
        exit(0)

    return img


def load_file(path):
    df = pd.read_csv(path)

    if "bbox_width" in df.columns or "bbox_height" in df.columns:
        df.rename({"bbox_x": "x1", "bbox_y": "y1"}, axis=1, inplace=True)
        df["x2"] = df["x1"] + df["bbox_width"]
        df["y2"] = df["y1"] + df["bbox_height"]

    df = df[["image_name", "x1", "y1", "x2", "y2"]].dropna()

    return [Sample(s[0], s[1], s[2], s[3], s[4]) for s in df.to_numpy()]


def load_bboxes(img_name, sample_array):
    return [sample.bbox for sample in sample_array if img_name == sample.image_name]


def update_window(val=0):
    img = load_image(img_names[val])

    estimated_bboxes = load_bboxes(img_names[val], detections)
    truth_bboxes = load_bboxes(img_names[val], truths)

    # red = ground truth
    # green = detection estimate

    for box in truth_bboxes:
        vision.draw_bbox(img, box, color=(0, 0, 255), line_width=2)

    for box in estimated_bboxes:
        vision.draw_bbox(img, box, color=(0, 255, 0))

    cv2.imshow(title, img)

    return


def main():
    global img_names, truths, detections

    img_names = listdir(args.dir)
    truths = load_file(args.labels)
    detections = load_file(args.detections)

    if len(img_names) == 0:
        print("No files found in given directory")
        exit(0)

    cv2.namedWindow(title)
    cv2.createTrackbar(img_index, title, 0, len(img_names) - 1, update_window)
    cv2.waitKey()

    return


if __name__ == "__main__":
    main()
