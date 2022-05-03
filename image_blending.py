import cv2
import numpy as np

def image_blending(canvas, mask_list) :

    gaussian_pyramid_list = []
    for i in range(len(canvas)) :
        gaussian_pyramid_list.append(get_gaussian_pyramid(canvas[i]))

    laplacian_pyramid_list = []
    for i in range(len(canvas)) :
        laplacian_pyramid_list.append(get_laplacian_pyramid(gaussian_pyramid_list[i]))

    mask_gaussian_pyramid_list = []
    for i in range(len(canvas)) :
        mask_gaussian_pyramid_list.append(get_gaussian_pyramid(mask_list[i]))

    blending_laplacian_pyramid = get_blending_laplacian_pyramid(laplacian_pyramid_list, mask_gaussian_pyramid_list)
    blending_image = reconstruct(blending_laplacian_pyramid)
    
    return blending_image

def reconstruct(blending_laplacian_pyramid) :

    blending_laplacian_pyramid = blending_laplacian_pyramid[::-1]
    recon_result = np.array([])
    
    for i in range(len(blending_laplacian_pyramid)-1) :

        if i == 0 :
            resize = image_upsampling(blending_laplacian_pyramid[i], blending_laplacian_pyramid[i+1].shape)
        else :
            resize = image_upsampling(recon_result, blending_laplacian_pyramid[i+1].shape)
        resize = np.float32(resize)
        recon_result = cv2.add(resize, np.float32(blending_laplacian_pyramid[i+1]))

    return recon_result

def get_blending_laplacian_pyramid(img_laplacian_pyramid_list, mask_gaussian_pyramid_list) :

    blending = []

    for pyramid_idx in range(len(img_laplacian_pyramid_list[0])) :
        unit = np.zeros_like(img_laplacian_pyramid_list[0][pyramid_idx])
        for img_idx in range(len(img_laplacian_pyramid_list)) :
            unit = unit + img_laplacian_pyramid_list[img_idx][pyramid_idx] * mask_gaussian_pyramid_list[img_idx][pyramid_idx]
        blending.append(unit)
    
    return blending

def get_laplacian_pyramid(gaussian_pyramid) :

    laplacian_pyramid = []
    gaussian_pyramid = gaussian_pyramid[::-1]

    for i in range(len(gaussian_pyramid)) :
        if i == 0 :
            laplacian_pyramid.append(gaussian_pyramid[i])
        else :
            gaussian_pyramid_n = np.float32(gaussian_pyramid[i])
            gaussian_pyramid_n_minus_1 = np.float32(image_upsampling(gaussian_pyramid[i-1], gaussian_pyramid_n.shape))
            laplacian_pyramid.append(cv2.subtract(gaussian_pyramid_n, gaussian_pyramid_n_minus_1))

    laplacian_pyramid = laplacian_pyramid[::-1]

    return laplacian_pyramid

def get_gaussian_pyramid(image) :

    
    gaussian_pyramid = []
    image = image.astype(np.float32)
    gaussian_pyramid.append(image)

    i = 0
    while len(gaussian_pyramid) < 5 :

        blurred = cv2.GaussianBlur(gaussian_pyramid[i], (5, 5), 1)

        blurred = image_downsampling(blurred)
        gaussian_pyramid.append(blurred)

        i += 1

    return gaussian_pyramid

def image_downsampling(image) :

    resized_image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    return resized_image

def image_upsampling(image, size) :

    resized_image = cv2.pyrUp(image)
    resized_image = cv2.resize(image, dsize=(size[1], size[0]))

    return resized_image