import cv2
import numpy as np

import seam_finder_DP as sfd

# 두 이미지에 대해서만 가능한 코드
# img_list는 warping 된 이미지여야 하고 미러링되면 안됨
def gain_based_esposure_compensation(img_list) :

    I = []
    N = []

    for row in range(len(img_list)) :

        I_row = []
        N_row = []
        print(str(row) + '번째 이미지의 밝기 계산 중')

        for column in range(len(img_list)) :
            
            overlap_mask, overlapBox = sfd.get_overlap_image(img_list[row], img_list[column])
            
            if overlapBox and row != column :
                I_row_column, N_row_column = calculate_mean_intensity_overlap(overlap_mask * img_list[row], overlap_mask)

                I_row.append(I_row_column)
                N_row.append(N_row_column)
            else :
                I_row.append(0)
                N_row.append(0)

        I.append(I_row)
        N.append(N_row)

    b = calculate_b_matrix(N)
    A = calculate_A_matrix(I, N, b)
    g = get_gain(A, b)

    # canvas1에 g[0], canvas2에 g[1]을 곱해서 사용
    return g 

def get_gain(A, b) :
    return np.dot(np.linalg.inv(A), b)

def calculate_A_matrix(I, N, b) :

    A = np.zeros((len(N), len(N))) 
    N = np.array(N)
    I = np.array(I)
    
    for row in range(len(N)) :
        for column in range(len(N)) :
            if row == column :
                sigma = 0
                for i in range(len(I[row])) :
                    sigma = sigma + I[row][i] ** 2 * N[row][i]
                A[row][column] = b[row] + 0.02 * sigma
            else :
                A[row][column] = -0.02 * I[row][column] * I[column][row] * N[row][column]
    
    return A

def calculate_b_matrix(N) :

    b = np.array([0]*len(N))

    for i in range(len(N)) :
        b[i] = 100 * np.sum(np.array(N[i]))

    print(b)
    return b

def calculate_mean_intensity_overlap(overlap, overlapMask) :

    intensity = 0
    N = 0

    for row in range(overlap.shape[0]) :
        for column in range(overlap.shape[1]) :
            if (overlapMask[row][column] != np.array([0, 0, 0])).any() :
                intensity = intensity + calculate_intensity(overlap[row][column])
                N = N + 1

    return intensity / N, N

def calculate_intensity(pixel) :
    return np.sqrt(pixel[0] ** 2 + pixel[1] ** 2 + pixel[2] ** 2) 