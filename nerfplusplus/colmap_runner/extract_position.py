
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_dir_path")
args = parser.parse_args()

with open(args.file_dir_path):
    