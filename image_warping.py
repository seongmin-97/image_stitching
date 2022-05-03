import cv2
import numpy as np

def image_warping(mirroring_list, H_matrix_list, boundingBox, borderMode) :
    
    canvas = []
    for i, image in enumerate(mirroring_list) :
        H_matrix_list[i] = H_matrix_list[i].astype('float32')
        warping = cv2.warpPerspective(image, H_matrix_list[i], (boundingBox[1]-boundingBox[0], boundingBox[3]-boundingBox[2]), borderMode=borderMode)
        canvas.append(warping)

    return canvas