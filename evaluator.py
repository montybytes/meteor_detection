"""
This script takes in the detections output and compares it to the ground truths
and displays the comparison metrics: precision, recall, F1 score and confusion matrix
"""

import argparse
import pandas as pd
import lib.vision as vision

from sample import Sample

# Setting up CLI argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--labels",
    default=None,
    required=True,
    help="The ground truth labels csv file",
)
parser.add_argument(
    "-p",
    "--predictions",
    default=None,
    required=True,
    help="The estimated detections csv file",
)
parser.add_argument(
    "-s",
    "--save_tp",
    default=False,
    action="store_true",
    help="Save true positives to file",
)

args = parser.parse_args()
# Flag to determine whether to save the true positives to disk
save_enabled = args.save_tp

# Reading the ground truth labels CSV into a pandas dataframe
labels = pd.read_csv(args.labels)

# Dropping unnecessary columns from the labels dataframe
labels.drop(columns=["label_name", "image_width", "image_height"], inplace=True)

# Renaming columns for bounding box coordinates
labels.rename({"bbox_x": "x1", "bbox_y": "y1"}, axis=1, inplace=True)

# Calculating x2 and y2 coordinates from x1, y1, bbox width, and height
labels["x2"] = labels["x1"] + labels["bbox_width"]
labels["y2"] = labels["y1"] + labels["bbox_height"]

# Dropping columns that are no longer needed
labels.drop(columns=["bbox_width", "bbox_height"], inplace=True)

# Reordering columns for consistency
labels = labels[["image_name", "x1", "y1", "x2", "y2"]]

# Reading the predictions CSV into a pandas dataframe
results = pd.read_csv(args.predictions)

# Converting the labels and predictions dataframes into lists of Sample objects
labels = [Sample(s[0], s[1], s[2], s[3], s[4]) for s in labels.to_numpy()]
results = [Sample(s[0], s[1], s[2], s[3], s[4]) for s in results.to_numpy()]

# Setting the IoU threshold for true positive detection
iou_threshold = 0.4

# Lists to store true positives and corresponding ground truth labels
true_positives = []
true_positives_labels = []

# Loop through each label in the ground truth dataset
for label_sample in labels:
    # Find all prediction samples that match the label's image name
    pred_samples = []
    for pred_sample in results:
        if pred_sample.image_name == label_sample.image_name:
            pred_samples.append(pred_sample)

    # Check IoU for each matching prediction sample against the label
    for pred_sample in pred_samples:
        iou = vision.get_IoU(pred_sample.bbox.points, label_sample.bbox.points)
        if iou > 0 and iou >= iou_threshold:
            true_positives.append(
                pred_sample
            )  # Add to true positives if IoU meets threshold
            true_positives_labels.append(label_sample)  # Mark the label as found

# Saving true positives to disk if the save flag is enabled
if save_enabled:
    with open("{}_{}.csv".format(args.predictions, "tp"), "a") as tp_file:
        for tp in true_positives:
            tp_file.write(
                "{},{},{},{},{}\n".format(
                    tp.image_name,
                    tp.bbox.x1,
                    tp.bbox.y1,
                    tp.bbox.x2,
                    tp.bbox.y2,
                )
            )

# Finding all ground truth samples that were not detected
false_negatives = [sample for sample in labels if sample not in true_positives_labels]


# Calculating the number of true positives, false negatives, and false positives
tp = len(true_positives)
fn = len(false_negatives)
fp = len(results) - tp

# Calculating precision, recall, and F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)

# Printing summary statistics
print(len(labels), " samples from labels")
print(len(results), " samples predicted")
print()
print("True Positives: {}".format(tp))
print("False Negatives: {}".format(fn))
print("False Positives: {}".format(fp))
print()
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1: {:.2f}".format(f1))
print()
