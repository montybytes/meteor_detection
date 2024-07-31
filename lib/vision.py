import cv2
import numpy as np

from shapes import Box, Line


def binarize(img, threshold, below=0, above=255):
    thresh, bim = cv2.threshold(img, threshold, above, cv2.THRESH_BINARY)
    bim[bim < below] = below

    return bim


def create_rectangle_mask(img, x, y, w, h):
    unwanted_region_mask = np.zeros(img.shape[:2], np.uint8)
    unwanted_region_mask[x:w, y:h] = 255

    return unwanted_region_mask


def morph(img, rows=1, cols=1, morph_type="open", iterations=1):
    # !throws an assertion error about the anchor but does not affect output
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
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength, maxLineGap)

    if lines is None:
        return []

    return [Line(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]


def draw_line(img, line: Line, color=(0, 255, 0), line_width=2):
    x1, y1, x2, y2 = line.points
    cv2.line(img, (x1, y1), (x2, y2), color, line_width)


def draw_lines(img, lines: list[Line], color):
    lines_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for line in lines:
        draw_line(lines_img, line, color)

    return lines_img


# todo: get more efficient way of doing this - don't create image
def get_bboxes(img, lines: list[Line], x_pad=8, y_pad=12) -> list[Box]:
    bboxes = []
    bbox_img = draw_lines(np.zeros(img.shape, np.uint8), lines, color=(255, 255, 255))

    contours, _ = cv2.findContours(
        cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append(Box(x - x_pad, y - y_pad, x + w + x_pad, y + h + y_pad))

    return bboxes


def draw_bbox(img, box: Box, color=(0, 255, 0), line_width=1):
    x1, y1, x2, y2 = box.points
    cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
    return


def skeletonize(img):
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
