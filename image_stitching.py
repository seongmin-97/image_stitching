import clindrical_warping as cw
import image_feature_matching as ifm
import estimate_homography as eh
import image_warping as iw
import gain_based_compensation as gbc
import image_blending as ib


import cv2
import math
import time
import numpy as np
from collections import deque

def image_stitching(fname_list, outputfname, NNDR=0.5, trial=500, compansation=True, intermediate_result=False) :

    img_list = read_image_list(fname_list)
    cyl_list, mirroring_cyl_list = cw.get_cylindrical_img_list(img_list)

    matching_graph = calculate_matching_graph(cyl_list, NNDR, trial)
    H_matrix_list, boundingBox, mask_list = all_matching_relation_homography(cyl_list, matching_graph, NNDR, trial, intermediate_result)

    print('start warping')
    
    mirrored_warping_image = iw.image_warping(mirroring_cyl_list, H_matrix_list, boundingBox, cv2.BORDER_REFLECT)

    if compansation :
        warping_image = iw.image_warping(cyl_list, H_matrix_list, boundingBox, cv2.BORDER_CONSTANT)

        print('start calculating gain')
        g = gbc.gain_based_esposure_compensation(warping_image)

        print('start apply gain')
        for i in range(len(g)) :
            mirrored_warping_image[i] = mirrored_warping_image[i] * g[i]

    print('start blending')
    start = time.time()
    blending_image = ib.image_blending(mirrored_warping_image, mask_list)
    print("blending time : ", time.time()-start)

    cv2.imwrite(outputfname, blending_image)


def all_matching_relation_homography(img_list, matching_graph, NNDR, trial, intermediate_result) :

    visited = [False] * len(img_list)
    visited[0] = True
    queue = deque([0])

    end_warping = []
    start = 0
    warped_image = []

    H_matrix_list = np.zeros((len(img_list), 3, 3))
    mask = []
    for i in range(len(img_list)) :
        mask.append([])

    i = 0
    while queue :
        img_pair = queue.popleft()

        # warping
        for pair in matching_graph[img_pair] :

            # 이미 와핑이 된 경우인지 확인 
            # ex) [0, 3]이 완료되었으면 [3, 0]은 안해도 된다.
            if [pair[1], pair[0]] in end_warping or pair[1] in warped_image :
                continue

            print(pair)
            if start == 0 :
                H_matrix, translate_matrix, bounding_Box, matching_pair = eh.estimate_homography(img_list[pair[0]], img_list[pair[1]], NNDR, trial)
                result, boundingBox, mask = eh.get_seam_and_mask(img_list[pair[0]], img_list[pair[1]], H_matrix, mask=mask, pair=pair, intermediate_result=intermediate_result, bundle=False, bounding_Box=bounding_Box, i=i)

                H_matrix_list[pair[0]] = np.array(H_matrix)
                H_matrix_list[pair[1]] = np.array(translate_matrix)

                warped_image.append(pair[0])
                warped_image.append(pair[1])
                start = 1
            else :
                H_matrix, translate_matrix, bounding_Box, matching_pair = eh.estimate_homography(img_list[pair[1]], result, NNDR, trial)
                result, boundingBox, mask = eh.get_seam_and_mask(img_list[pair[1]], result, H_matrix, mask=mask, pair=pair, intermediate_result=intermediate_result, bundle=False, bounding_Box=bounding_Box, i=i)

                for idx in warped_image :
                    H_matrix_list[idx] = translate_matrix.dot(H_matrix_list[idx])

                H_matrix_list[pair[1]] = np.array(H_matrix)

                warped_image.append(pair[1])

            end_warping.append(pair)
            i += 1
        # BFS
        for data in matching_graph[img_pair] :
            if not visited[data[1]] :
                queue.append(data[1])
                visited[data[1]] = True

    return H_matrix_list, boundingBox, mask

def calculate_matching_graph(img_list, NNDR, trial) :

    matching_graph = []
    n_f, n_i = calculate_number_of_feature_and_H(img_list, NNDR, trial)

    for i in range(len(img_list)) :
        node_info = []

        for j in range(len(img_list)) :
            if isMatching(n_f[i][j], n_i[i][j]) :
                node_info.append([i, j])

        matching_graph.append(node_info)

    print("matching_graph : ", matching_graph)
        
    return matching_graph

def calculate_number_of_feature_and_H(img_list, NNDR, trial) :
    
    n_f = []
    n_i = []

    for row in range(len(img_list)) :
        
        num_matching = []
        num_inlier = []

        for column in range(len(img_list)) :
            
            if row == column :
                num_matching.append(0)
                num_inlier.append(0)
            elif row > column :
                num_matching.append(n_f[column][row])
                num_inlier.append(n_i[column][row])
            else :
                matching_keypoint1, matching_keypoint2, number_of_matching = ifm.get_matching_feature(img_list[row], img_list[column], NNDR)
                
                if number_of_matching < 4 :
                    num_matching.append(number_of_matching)
                    num_inlier.append(0)
                else :
                    _, best_inlier_num = eh.RANSAC(matching_keypoint1, matching_keypoint2, len(matching_keypoint1), trial)
                    num_matching.append(number_of_matching)
                    num_inlier.append(best_inlier_num)

        n_f.append(num_matching)
        n_i.append(num_inlier)

        print("피쳐 개수, H 계산 중...", (row+1)/len(img_list))

    return n_f, n_i

def isMatching(n_f, n_i) :
    
    p1 = 0.6 # 
    p0 = 0.1
    p_m1 = 0.000001
    p_m0 = 1 - p_m1
    p_min = 0.999

    f_when_p_m1 = math.comb(n_f, n_i) * (p1 ** n_i) * ((1 - p1) ** (n_f - n_i))
    f_when_p_m0 = math.comb(n_f, n_i) * (p0 ** n_i) * ((1 - p0) ** (n_f - n_i))

    matching_probability = 1.0 / (1 + (f_when_p_m0 * p_m0) / (f_when_p_m1 * p_m1 + 0.000000000000001))

    if matching_probability > p_min :
        return True
    else :
        return False

def read_image_list(fname_list) :

    img_list = []

    for fname in fname_list :
        img_list.append(cv2.imread(fname))

    return img_list
# 
# fname_list = ['./data1/IMG_0423.jpg', './data1/IMG_0424.jpg', './data1/IMG_0422.jpg', './data1/IMG_0425.jpg', './data1/IMG_0421.jpg', './data1/IMG_0426.jpg', './data1/IMG_0420.jpg', './data1/IMG_0427.jpg']
# fname_list = ['./data1/IMG_0423.jpg', './data1/IMG_0420.jpg']
# fname_list = ['./data/museum1.jpg', './data/museum3.jpg', './data/museum2.jpg', './data/museum4.jpg', './data/museum5.jpg']
# fname_list = ['./data/school2.jpg', './data/school1.jpg', './data/school3.jpg', './data/school4.jpg', './data/school5.jpg', './data/school6.jpg', './data/school8.jpg']
# image_stitching(fname_list, './school_result_no_gain_no_ba.jpg', NNDR=0.7, trial=500, compansation=False, intermediate_result=True)
# fname_list = ['./data/sample.jpg', './data/sample1.jpg']
# fname_list = ['lab1.jpg', 'lab2.jpg']
# image_stitching(fname_list, './sample_result.jpg', NNDR=0.7, trial=500, compansation=True)
# fname_list = ['./data/museum5.jpg', './output1.jpg']
# image_stitching(fname_list, './output2.jpg', NNDR=0.7, trial=500)