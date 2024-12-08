"""
This script performs McNemar's Z test on the output predictions on two result
files generated from the evaluation script. The files must only contain the 
true positives from the detection process. The evaulator.py script provides an
option for saving these values to disk.
"""

import numpy
import argparse
import pandas as pd
import lib.vision as vision

# Setting up CLI parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "fileA",
    default=None,
    help="The files to compare",
)
parser.add_argument(
    "fileB",
    default=None,
    help="The files to compare",
)

args = parser.parse_args()

# Reading CSV files into pandas dataframes
results_a = pd.read_csv(args.fileA, header=None)
results_b = pd.read_csv(args.fileB, header=None)

# Converting dataframes into lists of tuples for comparison
results_a = [(s[0], s[1], s[2], s[3], s[4]) for s in results_a.to_numpy()]
results_b = [(s[0], s[1], s[2], s[3], s[4]) for s in results_b.to_numpy()]

# Counting the number of elements in results_a that are not in results_b
Nsf = len([result for result in results_a if result not in results_b])

# Counting the number of elements in results_b that are not in results_a
Nfs = len([result for result in results_b if result not in results_a])

# Calculating the Z-score to measure the difference between the two sets
Z = numpy.sqrt(numpy.square(numpy.abs(Nsf - Nfs) - 1) / (Nsf + Nfs))

# Printing the Z-score
print("Nsf:{}\nNfs:{}\n\nZ:{}".format(Nsf, Nfs, Z))
