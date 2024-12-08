import cv2
import numpy as np

from shapes import Box, Line


def load_image(path, mode=cv2.IMREAD_COLOR):
    """Routine to wrap the OpenCV image loading process with error handling

    Args:
        path: path of image on disk
        mode: Open CV image reading flag
    """

    img = cv2.imread(path, mode)

    if img is None:
        print("Error: Could not open or find the image: ", path)
        exit(0)

    return img


def to_color(img):
    """Routine to wrap OpenCV grayscale to color converter"""

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def to_gray(img):
    """Routine to wrap OpenCV color to grayscale converter"""

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(img, threshold, below=0, above=255, adaptive=True):
    """Routine to create a binarized image by thresholding values and setting 
    larger values to the `above` param and values lower than to `below` param
    
    Args:
        img: source image
        threshold: binarizing threshold
        below: value for pixels less than threshold
        above: value for pixels greater than threshold
        adaptive: flag to use adaptive Otsu thresholding

    Returns:
        bim: binarized image
    """

    bim = None
    method = cv2.THRESH_BINARY

    if adaptive:
        img = cv2.GaussianBlur(img, (3, 3))
        method = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    thresh, bim = cv2.threshold(img, threshold, above, method)
    bim[bim < below] = below

    return bim


def create_rectangle_mask(img, x, y, w, h):
    """Routine to create a rectangular image mask given the rectangle's top left
    corner coordinates and its width and height

    Args:
        img: source image
        x: top left x-coordinate
        y: top left y-coordinate
        w: rectangle width
        h: rectangle height

    Returns:
        mask: array representing mask area
    """

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[x:w, y:h] = 255

    return mask


def morph(img, rows=1, cols=1, morph_type="open", iterations=1):
    """Routine to perform basic morphology such as erosion, dilation, opening 
    and closing.

    Args:
        img: source image
        rows: kernel size rows
        cols: kernel size columns
        morph_type: morphological operation to perform
        iterations: number of times to perform operation

    Returns:
        final: image with operations applied to
    """
    
    # open-cv function that assists in creating a kernel with a given shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols, rows))

    final = img.copy()

    for _ in range(iterations):
        if morph_type == "erode":
            final = cv2.erode(final, kernel)

        if morph_type == "dilate":
            final = cv2.dilate(final, kernel)

        if morph_type == "open":
            final = cv2.dilate(cv2.erode(final, kernel), kernel)

        if morph_type == "close":
            final = cv2.erode(cv2.dilate(final, kernel), kernel)

    return final


def get_lines(img, rho, theta, threshold, minLineLength, maxLineGap) -> list[Line]:
    """Routine to detect lines on an image using the probabilistic variant of 
    Hough's Line transform.

    Args:
        img: source image
        rho: distance of line resolution
        theta: the angle of lines
        morph_type: morphological operation to perform
        iterations: number of times to perform operation

    Returns:
        final: image with operations applied to
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength, maxLineGap)

    if lines is None:
        return []

    return [Line(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]


def draw_line(img, line: Line, color=(0, 255, 0), line_width=2):
    """Routine to draw a line on an image.

    Args:
        img: source image
        line: line object containing points
        color: color of line
        line_width: width of line
    """
     
    x1, y1, x2, y2 = line.points
    cv2.line(img, (x1, y1), (x2, y2), color, line_width)


def draw_lines(img, lines: list[Line], color):
    """Routine to draw lines on an image.

    Args:
        img: source image
        lines: list of line object containing points
        color: color to draw on lines
    """
    
    lines_img = to_color(img.copy())

    for line in lines:
        draw_line(lines_img, line, color)

    return lines_img


# todo: get more efficient way of doing this - don't create image
def get_bboxes(img, lines: list[Line], x_pad=8, y_pad=12) -> list[Box]:
    """Routine to find bounding boxes of an image by finding the contours of
    the lines provided.

    Args:
        img: source image
        lines: list of line object containing points
        x_pad: horizontal padding of bounding box
        y_pad: vertical padding of bounding box

    Returns:
        bboxes: detected bounding boxes
    """

    bboxes = []
    bbox_img = draw_lines(np.zeros(img.shape, np.uint8), lines, color=(255, 255, 255))

    contours, _ = cv2.findContours(
        to_gray(bbox_img),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append(Box(x - x_pad, y - y_pad, x + w + x_pad, y + h + y_pad))

    return bboxes


def draw_bbox(img, box: Box, color=(0, 255, 0), line_width=1):
    """Routine to draw bounding box on an image.

    Args:
        img: source image
        box: box object containing points of the box
        color: color to draw box
        line_width: width of line of the box
    """

    x1, y1, x2, y2 = box.points
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
    return


def skeletonize(img):
    """Routine to perform skeletonization/thinning on an binary image.

    Args:
        img: source image
    
    Returns:
        skeleton: the binary image of the skeleton of the features in the source
            image
    """

    skeleton = np.zeros(img.shape, np.uint8)

    rows, cols = (3, 3)

    while True:
        eroded = morph(img, rows=rows, cols=cols, morph_type="erode")
        opened = morph(eroded, rows=rows, cols=cols, morph_type="dilate")

        opened = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, opened)

        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def get_area(x1, y1, x2, y2):
    """Routine to find the area of a bounding box.

    Args:
        x1, y1, x2, y2: coordinates of the box

    Returns:
        area of the box
    """

    return abs(x2 - x1) * abs(y2 - y1)


def is_intersecting(bbox_a, bbox_b):
    """Routine to check if two boxes are intersecting.

    Args:
        bbox_a, bbox_b: the boxes to be checked

    Returns:
        boolean value that determines if the boxes are intersecting or not
    """

    a_x1, a_y1, a_x2, a_y2 = bbox_a
    b_x1, b_y1, b_x2, b_y2 = bbox_b

    return not (a_x2 < b_x1 or b_x2 < a_x1 or a_y2 < b_y1 or b_y2 < a_y1)


def get_intersection_area(bbox_a, bbox_b):
    """Routine to find the intersection area of two bounding boxes.

    Args:
        bbox_a, bbox_b: the boxes to find the overlapping area of

    Returns:
        area of the intersection
    """

    if not is_intersecting(bbox_a, bbox_b):
        return 0

    a_x1, a_y1, a_x2, a_y2 = bbox_a
    b_x1, b_y1, b_x2, b_y2 = bbox_b

    int_x1, int_y1 = (max(a_x1, b_x1), max(a_y1, b_y1))
    int_x2, int_y2 = (min(a_x2, b_x2), min(a_y2, b_y2))

    return get_area(int_x1, int_y1, int_x2, int_y2)


def get_union_area(bbox_a, bbox_b):
    """Routine to find the area of the union of two bounding boxes.

    Args:
        bbox_a, bbox_b: the boxes to find the overlapping area of

    Returns:
        area of the union of the two boxes
    """

    a_x1, a_y1, a_x2, a_y2 = bbox_a
    b_x1, b_y1, b_x2, b_y2 = bbox_b

    un_x1, un_y1 = (min(a_x1, b_x1), min(a_y1, b_y1))
    un_x2, un_y2 = (max(a_x2, b_x2), max(a_y2, b_y2))

    return get_area(un_x1, un_y1, un_x2, un_y2)


def get_IoU(bbox_a, bbox_b):
    """Routine to find the ratio of intersection and union of two bounding boxes.

    Args:
        bbox_a, bbox_b: the boxes to find the IoU

    Returns:
        IoU ratio of the bounding boxes
    """

    return get_intersection_area(bbox_a, bbox_b) / get_union_area(bbox_a, bbox_b)
