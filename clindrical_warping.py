import cv2
import math
import numpy as np

# https://www.codetd.com/en/article/12557387
def cylindrical_projection(img) :
    rows = img.shape[0]
    cols = img.shape[1]

    #f = cols / (2 * math.tan(np.pi / 8))
    result = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    alpha = math.pi / 4
    f = cols / (2 * math.tan(alpha / 2))
    x_list, y_list = np.indices((img.shape[0], img.shape[1]), dtype=np.float32)

    for  y in range(rows):
        for x in range(cols):
            theta = math.atan((x- center_x )/ f)
            point_x = f * math.tan( (x-center_x) / f) + center_x
            point_y = (y-center_y) / math.cos(theta) + center_y
            x_list[y][x] = point_x
            y_list[y][x] = point_y
            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                result[y , x, :] = img[int(point_y) , int(point_x), :]
    
    test = cv2.remap(img, x_list, y_list, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return result, test

def get_cylindrical_img_list(img_list) :

    cyl_list = []
    mirroring_cyl_list = []
    
    for image in img_list :
        normal, mirroring = cylindrical_projection(image)
        cyl_list.append(normal)
        mirroring_cyl_list.append(mirroring)

    return cyl_list, mirroring_cyl_list