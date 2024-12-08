"""
This python script performs detection on an image or set of images and allows 
for calibration of the detection parameters. The output consists of a set of 
images showing the results at important intervals of exectuion. There are also 
sliders that allow for calibration of values that influence the detection. If 
the save flag has been passed, the program will save these values upon 
termination.
"""

import cv2
import json
import numpy as np
import argparse

from os import listdir
from os.path import isfile, join

import lib.vision as vision
import lib.utils as utils

# window display boilerplate
title = "Simple Meteor Detection: Calibrator"

img_index = "Image Index:"
binary_thresh = "Bin Thresh:"
row_count = "Row count:"
hough_thresh = "Hgh Thresh:"
line_length = "Hgh Len:"
line_gap = "Hgh Gap:"
bb_thresh = "BB Thresh:"

# the single image path to be used for calibration
img_path = None
# the image paths loaded to be used for calibration
img_paths = []

# CLI setup
parser = argparse.ArgumentParser()

parser.add_argument(
    "file",
    nargs="?",
    default=None,
    help="The image file to display during calibration",
)
parser.add_argument(
    "--dir",
    default=None,
    nargs="?",
    help="The folder to load calibration images from",
)
parser.add_argument(
    "-s",
    "--save",
    default=False,
    action="store_true",
    help="Save config to file",
)

args = parser.parse_args()
# save flag
save_mode = args.save


def get_image():
    """Return an image based on calibration mode; single image or image set"""

    if len(img_paths) > 0:
        # select image path from image path set using trackbar position
        index = cv2.getTrackbarPos(img_index, title)
        print("current image:", img_paths[index])
        return vision.load_image(img_paths[index], cv2.IMREAD_GRAYSCALE)
    else:
        return vision.load_image(img_path, cv2.IMREAD_GRAYSCALE)


def detect_broadbands(bboxes):
    """Filter out boxes that exist within the threshold of other boxes on the
    same X-axis value"""

    broadbands = []

    for box in bboxes:
        # if box already exists in list, skip it
        if box in broadbands:
            print("skipping already detected broadband")
            continue

        x, y = box.midpoint
        # list to store all boxes that are in range of the current box's X-position
        matches = []

        for i in range(bboxes.index(box) + 1, len(bboxes)):
            _x, _y = bboxes[i].midpoint
            # threshold calculation from current box's midpoint X value
            _thresh = cv2.getTrackbarPos(bb_thresh, title)
            _min, _max = (x - _thresh, x + _thresh)

            # check if centers are within current midpoint's threshold
            if _min <= _x <= _max:
                matches.append(bboxes[i])

        # add current box to matches since if it is a broadband, it is included in the boxes of broadbands
        matches.insert(0, box)

        # if the list has more than the current box, it is considered a broadband
        if len(matches) > 1:
            broadbands.append(matches)

    return broadbands


def add_text(img, text, position):
    """Simple routine to display text at a given position on an image"""
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 1
    color = (255, 255, 255)

    thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(img, text, position, font, font_scale, color, thickness, line_type)
    return


def vertical_extraction(val=0):
    """Routine to perform the detection process and update image on parameter
    change

    Args:
        none

    Returns:
        detections: list of predicted bounding boxes
    """
    img = get_image()

    # removing legends and axes (unimportant areas)
    mask = vision.create_rectangle_mask(img, 64, 64, 607, 927)
    masked = cv2.bitwise_or(img, img, mask=mask)

    # binarize image
    thresh = cv2.getTrackbarPos(binary_thresh, title)
    binarized = vision.binarize(masked, thresh, adaptive=False)

    rows = cv2.getTrackbarPos(row_count, title)

    # performing vertical erosion & dilation
    vertical_features = vision.morph(binarized, rows=rows, morph_type="erode")

    # skeletonization pass
    skeleton = vision.skeletonize(vertical_features)
    # close vertical holes in skeleton
    skeleton = vision.morph(skeleton, rows=20, cols=2, morph_type="close")
    # remove more vertical noise
    skeleton = vision.morph(skeleton, rows=15, cols=1, morph_type="open", iterations=2)

    # line detection using probabilistic Hough Transform
    lines = vision.get_lines(
        skeleton,
        rho=1,  # 1 pixel distance check
        theta=np.pi / 180,  # equivalent to 1 degree
        threshold=cv2.getTrackbarPos(hough_thresh, title),
        minLineLength=cv2.getTrackbarPos(line_length, title),
        maxLineGap=cv2.getTrackbarPos(line_gap, title),
    )

    # getting bounding boxes for the lines detected
    bboxes = vision.get_bboxes(img, lines, x_pad=4)

    # for each box, get the center and find other boxes in the same list that have a center +/- 2px off  on the x-axis
    boxes_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_w, img_h = (927 - 64, 607 - 64)
    img_center_y = (img_h // 2) + 64

    # flattening bounding boxes array into a 1-dimensional array for fast looping
    broadbands = utils.flatten(detect_broadbands(bboxes))

    # removing broadband signals from the true detection predictions
    detections = [bbox for bbox in bboxes if bbox not in broadbands]

    # loop to draw the detected bounding boxes and display with color based on distance from center of image
    for box in detections:
        mid_x, mid_y = box.midpoint
        deviation = abs(mid_y - img_center_y)
        deviation = utils.mapFromTo(deviation, 0, img_h / 2, 0, 1)
        vision.draw_bbox(boxes_img, box, (0, 255 - 255 * deviation, 255 * deviation))

    # merging the images output from different steps to provide detailed output
    row_1 = vision.to_color(cv2.hconcat([binarized, vertical_features]))
    row_2 = cv2.hconcat([vision.to_color(skeleton), boxes_img])
    final = cv2.vconcat([row_1, row_2])

    # writing texts on final image to label the different steps
    add_text(final, "binarized", position=(0, 24))
    add_text(final, "vertical erosion", position=(img.shape[1], 24))
    add_text(final, "skeletonized", position=(0, img.shape[0] + 24))
    add_text(final, "bounding boxes", position=(img.shape[1], img.shape[0] + 24))

    cv2.imshow(title, final)
    return


def main():
    global img_path, img_paths

    cv2.namedWindow(title)

    # reading file names from directory argument
    if args.dir is not None:
        files = listdir(args.dir)

        print(len(files))

        if len(files) == 0:
            print("No files found in given directory")
            exit(0)

        # storing file names in global array
        img_paths = [join(args.dir, f) for f in files if isfile(join(args.dir, f))]
        cv2.createTrackbar(img_index, title, 0, len(img_paths) - 1, vertical_extraction)

    else:
        img_path = args.file

    # parameter sliders setup
    cv2.createTrackbar(binary_thresh, title, 14, 255, vertical_extraction)
    cv2.createTrackbar(row_count, title, 12, 32, vertical_extraction)
    cv2.createTrackbar(hough_thresh, title, 20, 100, vertical_extraction)
    cv2.createTrackbar(line_length, title, 50, 100, vertical_extraction)
    cv2.createTrackbar(line_gap, title, 5, 100, vertical_extraction)
    cv2.createTrackbar(bb_thresh, title, 1, 80, vertical_extraction)

    cv2.setTrackbarMin(row_count, title, 1)
    cv2.setTrackbarMin(bb_thresh, title, 1)

    vertical_extraction()

    cv2.waitKey()

    # if save flag is enabled, perform writing to disk of final detection parameters
    if save_mode:
        # save parameters to json file
        with open("config.json", "w") as cfg_file:
            # program parameters
            data = {
                "binary_threshold": cv2.getTrackbarPos(binary_thresh, title),
                "row_count": cv2.getTrackbarPos(row_count, title),
                "hough_threshold": cv2.getTrackbarPos(hough_thresh, title),
                "hough_line_length": cv2.getTrackbarPos(line_length, title),
                "hough_line_gap": cv2.getTrackbarPos(line_gap, title),
                "broadband_thresh": cv2.getTrackbarPos(bb_thresh, title),
            }
            json.dump(data, cfg_file, indent=2)
            print("--\nconfiguration saved to file\n--")

    return


if __name__ == "__main__":
    main()
