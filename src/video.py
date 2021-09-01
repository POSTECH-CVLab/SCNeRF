import cv2
import numpy as np
import argparse
import os

def convert_video(args):
    arr = []
    for file in sorted(os.listdir(args.dir)):
        if "fg" in file or "bg" in file or not (file.endswith(".png") or file.endswith(".jpg")):
            continue
        img_path = os.path.join(args.dir, file)
        arr.append(cv2.imread(img_path))
    h, w, c = arr[-1].shape
    size  = (w, h)
    out = cv2.VideoWriter(os.path.join(args.dir, ".." ,"example.avi"), cv2.VideoWriter_fourcc(*"DIVX"), 30, size)

    for i in range(len(arr)):
        out.write(arr[i])
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    convert_video(args)