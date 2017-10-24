# Imports
import argparse

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run emoji prediction classifiers and output accuracy results.")
parser.add_argument('-c', '--classifier', nargs='+', default='all', choices=['nb', 'lr'],
    help='specifies which classifier(s) to use (default: %(default)s) possible classifiers: %(choices)s',
    metavar='C', dest='classifier_type')
parser.add_argument('-d', '--data', nargs=1, default='train',
    help='name of dataset file to use (must be in Data directory)',
    metavar='dataset', dest='data')
parser.add_argument('-n', nargs=1,
    help='number of data entries to train/evaluate on',
    metavar='N', dest='num_instances')
args = parser.parse_args()
print(args)

# Load Data

# Extract Features

# Instantiate Classifiers

# Train Classifiers on Extracted Features

# Evaluate Classifiers

# Graphing
