from __future__ import print_function

import os

import numpy as np
import cv2 as cv

import argparse
import sys

import image_stitching as stitching

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('dir', help = 'input images directory')

def load_images_from_dir(dir):
    fname = []
    for filename in sorted(os.listdir(dir)):
        fname.append(dir + '/' + filename)
    return fname

def main():
    args = parser.parse_args()

    fname_list = load_images_from_dir(args.dir)
    
    stitching.image_stitching(fname_list, args.dir+'.jpg', NNDR=0.7, trial=500)

if __name__ == '__main__':
    main()