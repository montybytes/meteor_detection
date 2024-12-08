"""
This python script performs only detection on an image or set of images using 
the calibrated output from calibrator.py. The output consists of a comma-separated
list of the bounding boxes which can then be piped into a file via the terminal.
"""

import cv2
import json
import argparse
import numpy as np

from os import listdir
from os.path import isfile, join

import lib.vision as vision
import lib.utils as utils

# setting up command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--display",
    default=False,
    action="store_true",
    help="Display the image and detections (default:false)",
)
parser.add_argument(
    "-c",
    "--config",
    default="config.json",
    help="The configuration json file",
)
parser.add_argument(
    "--dir",
    default=None,
    help="The folder to perform detection on",
)
parser.add_argument(
    "file",
    nargs="?",
    default=None,
    help="The file to perform detection on",
)

args = parser.parse_args()
# flag to display final detections on the image if a single image is passed to the script
display_mode = args.display


def load_parameters(path):
    """Routine to load calibration parameters from disk

    Args:
        path: path to the configured parameters

    Returns:
        json object containing the parameters
    """
    try:
        with open(path) as params_file:
            return json.load(params_file)
    except:
        print("Error: An error was encountered trying to load config file")
        exit(0)


def detect_broadbands(bboxes, threshold=5):
    broadbands = []

    for box in bboxes:
        if box in broadbands:
            continue

        x, y = box.midpoint
        matches = []

        for i in range(bboxes.index(box) + 1, len(bboxes)):
            _x, _y = bboxes[i].midpoint

            # check if centers are within range
            range_min, range_max = (x - threshold, x + threshold)
            if range_min <= _x <= range_max:
                matches.append(bboxes[i])

        matches.insert(0, box)

        if len(matches) > 1:
            broadbands.append(matches)

    return broadbands


def detect_meteors(img, params):
    """Stripped down routine equivalent to routine in 
     `calibrator.py` named vertical_extraction to perform the detection process

    Args:
        img: source image
        params: calibrated/configured detection parameters

    Returns:
        detections: list of predicted bounding boxes
    """

    # remove legends and axes (unimportant areas)
    mask = vision.create_rectangle_mask(img, 64, 64, 607, 927)
    masked = cv2.bitwise_or(img, img, mask=mask)

    # binarize image
    binarized = vision.binarize(masked, params["binary_threshold"], adaptive=False)

    # vertical erosion & dilation
    vertical_features = vision.morph(
        binarized, rows=params["row_count"], morph_type="open"
    )
    # remove vertical noise
    vertical_features = vision.morph(
        vertical_features, rows=params["row_count"], morph_type="erode"
    )

    # skeletonize the vertical features to get vertical lines
    skeleton = vision.skeletonize(vertical_features)

    # close vertical holes in skeleton
    skeleton = vision.morph(skeleton, rows=20, cols=2, morph_type="close")

    # remove more vertical noise
    skeleton = vision.morph(skeleton, rows=15, cols=1, morph_type="open", iterations=2)

    # detect lines from skeleton using probabilistic Hough Transform
    lines = vision.get_lines(
        skeleton,
        rho=1,  # 1 pixel distance check
        theta=np.pi / 180,  # equivalent to 1 degree
        threshold=params["hough_threshold"],
        minLineLength=params["hough_line_length"],
        maxLineGap=params["hough_line_gap"],
    )

    # getting bounding boxes for the lines detected
    bboxes = vision.get_bboxes(img, lines, x_pad=4)

    # flattening bounding boxes array into a 1-dimensional array for fast looping
    broadbands = utils.flatten(detect_broadbands(bboxes, params["broadband_thresh"]))

    # removing broadband signals from the true detection predictions
    detections = [bbox for bbox in bboxes if bbox not in broadbands]

    return detections


def display_detections(img, detections):
    """Routine to display the detections if required

    Args:
        img: source image
        detections: list of bounding boxes
    """

    boxes_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_h = 607 - 64
    img_center_y = (img_h // 2) + 64

    for box in detections:
        mid_x, mid_y = box.midpoint
        deviation = abs(mid_y - img_center_y)
        deviation = utils.mapFromTo(deviation, 0, img_h / 2, 0, 1)
        vision.draw_bbox(boxes_img, box, (0, 255 - 255 * deviation, 255 * deviation))

    cv2.imshow("Simple Meteor Detection: Detector(View Mode)", boxes_img)
    cv2.waitKey()


def main():
    # load detection configurations
    params = load_parameters(args.config)

    # load images
    if args.dir is not None:
        files = listdir(args.dir)

        if len(files) == 0:
            print("No files found in given directory")
            exit(0)

        imgs_dict = {}

        for f in files:
            path = join(args.dir, f)
            if isfile(path):
                imgs_dict[f] = path

        # printing the output of the detected bounding boxes in csv format
        print("image_name,x1,y1,x2,y2")
        for key, path in imgs_dict.items():
            img = vision.load_image(path, cv2.IMREAD_GRAYSCALE)
            detections = detect_meteors(img, params)
            if len(detections) == 0:
                continue
            else:
                for box in detections:
                    x1, y1, x2, y2 = box.points
                    print("{},{},{},{},{}".format(key, x1, y1, x2, y2))

    else:
        img = vision.load_image(args.file, cv2.IMREAD_GRAYSCALE)
        detections = detect_meteors(img, params)
        if len(detections) != 0:
            for box in detections:
                x1, y1, x2, y2 = box.points
                print("{},{},{},{}".format(x1, y1, x2, y2))

    # display image with predicted outputs if display flag is passed
    if display_mode:
        display_detections(img, detections)

    return


if __name__ == "__main__":
    main()
