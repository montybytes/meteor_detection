import cv2
import json
import numpy as np

import lib.vision as vision
import lib.utils as utils


# ? plans:
# [-] check if blurring & denoising help edge detection
# [-] use erosion and dilation to get vertical features
# [-] use greyscale image
# [-] try to use skeletonizing on vertical features
# [-] weed out non-vertical detected lines and boundary cases e.g broadband signals
# [ ] use a center line biasing method to assign confidence values to detected lines
# [-] develop a pipleine to perform all steps on single image
# [ ] evaluate performance using fact.py | create evaluation script

title = "Simple Meteor Detection"

img_index = "Image Index:"
binary_thresh = "Bin Thresh:"
row_count = "Row count:"
hough_thresh = "Hgh Thresh:"
line_length = "Hgh Len:"
line_gap = "Hgh Gap:"
bb_thresh = "BB Thresh:"

imgs = []

lines = []


def get_img(window):
    # change image undergoing processing to visualize the parameters on different images
    index = cv2.getTrackbarPos(img_index, window)
    return imgs[index]


def detect_broadbands(bboxes):
    broadbands = []

    for box in bboxes:
        if box in broadbands:
            print("skipping already detected broadband")
            continue

        x, y = box.midpoint
        matches = []

        for i in range(bboxes.index(box) + 1, len(bboxes)):
            _x, _y = bboxes[i].midpoint
            # check if centers are within current midpoint's threshold
            _thresh = cv2.getTrackbarPos(bb_thresh, title)
            _min, _max = (x - _thresh, x + _thresh)

            if _min <= _x <= _max:
                matches.append(bboxes[i])

        matches.insert(0, box)

        if len(matches) > 1:
            broadbands.append(matches)

    return broadbands


def vertical_extraction(val=0):
    # start_time = timeit.default_timer()

    global title, imgs

    img = get_img(title)

    # *pass: removing legends and axes (unimportant areas)
    mask = vision.create_rectangle_mask(img, 64, 64, 607, 927)
    masked = cv2.bitwise_or(img, img, mask=mask)

    # *pass: binarize image
    thresh = cv2.getTrackbarPos(binary_thresh, title)
    binarized = vision.binarize(masked, thresh)

    # # *pass(2): performing vertical erosion & dilation
    rows = cv2.getTrackbarPos(row_count, title)

    vertical_features = vision.morph(binarized, rows=rows, morph_type="open")
    # remove vertical noise
    vertical_features = vision.morph(vertical_features, rows=rows, morph_type="erode")

    # *pass: skeletonization
    skeleton = vision.skeletonize(vertical_features)
    # close vertical holes in skeleton
    skeleton = vision.morph(skeleton, rows=20, cols=2, morph_type="close")
    # remove more vertical noise
    skeleton = vision.morph(skeleton, rows=15, cols=1, morph_type="open", iterations=2)

    # *pass: line detection using probabilitic Hough Transform
    lines = vision.get_lines(
        skeleton,
        rho=1,  # 1 pixel distance check
        theta=np.pi / 180,  # equivalent to 1 degree
        threshold=cv2.getTrackbarPos(hough_thresh, title),
        minLineLength=cv2.getTrackbarPos(line_length, title),
        maxLineGap=cv2.getTrackbarPos(line_gap, title),
    )

    # *pass: drawing lines on image - useful for display but not necessary for final program
    bboxes = vision.get_bboxes(img, lines, x_pad=4)

    # for each box, get the center and find other boxes in the same list that have a center +/- 2px off  on the x-axis
    boxes_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_w, img_h = (927 - 64, 607 - 64)
    img_center_y = (img_h // 2) + 64

    cv2.line(boxes_img, (64, img_center_y), (img_w + 64, img_center_y), (0, 0, 255), 2)

    # !!
    broadbands = utils.flatten(detect_broadbands(bboxes))

    detections = [bbox for bbox in bboxes if bbox not in broadbands]

    for box in detections:
        # calculate midpoints
        mid_x, mid_y = box.midpoint
        deviation = abs(mid_y - img_center_y)
        deviation = utils.mapFromTo(deviation, 0, img_h / 2, 0, 1)
        vision.draw_bbox(boxes_img, box, (0, 255 - 255 * deviation, 255 * deviation))

    # final_detections = vision.draw_bboxes(img, detections)

    # extra addon: make script save bounding boxes to file and also draw boxes on original image and save to folder

    # end_time = timeit.default_timer() - start_time
    # print("Executed in: ", end_time * 1000, "ms")

    cv2.imshow(title, boxes_img)
    return


def main():
    # -- images to test with --
    img_paths = [
        "dataset/images/train/BEBILZ_SYS001-2024-05-25-03-00.png",  # (only simple + complex meteors)
        "dataset/images/train/BEBILZ_SYS001-2024-05-25-06-00.png",  # (many planes)
        "dataset/images/train/BEBILZ_SYS001-2024-05-25-12-00.png",  # (many planes + complex meteor)
        "dataset/images/train/BEBOEC_SYS001-2024-05-25-01-00.png",  # (very noisy)
        "dataset/images/train/BEDINA_SYS001-2024-05-25-13-00.png",  # (no noise)
        "dataset/images/test/BEBILZ_SYS001-2024-05-26-14-00.png",  # (many broadband signals)
    ]

    # loading all the images
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Could not open or find the image: ", path)
            exit(0)

        imgs.append(img)

    # creating configuration ui
    cv2.namedWindow(title)
    cv2.createTrackbar(img_index, title, 0, len(imgs) - 1, vertical_extraction)
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

    # program parameters
    data = {
        "binary_threshold": cv2.getTrackbarPos(binary_thresh, title),
        "row_count": cv2.getTrackbarPos(row_count, title),
        "hough_threshold": cv2.getTrackbarPos(hough_thresh, title),
        "hough_line_length": cv2.getTrackbarPos(line_length, title),
        "hough_line_gap": cv2.getTrackbarPos(line_gap, title),
        "broadband_thresh": cv2.getTrackbarPos(bb_thresh, title),
    }

    # save parameters to json file
    with open("config.json", "w") as cfg_file:
        json.dump(data, cfg_file, indent=2)

    return


if __name__ == "__main__":
    main()
