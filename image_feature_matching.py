from turtle import distance
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_feature_SIFT(img1, showImg) :
    # 이미지 불러오기
    img1 = img1.astype(np.uint8)
    # 색상변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # SIFT
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    # 출력
    if showImg :
        plt.imshow(img1)
        for i in range(len(kp1)) :
            plt.scatter(kp1[i].pt[0], kp1[i].pt[1], c='red', s=0.5)
        plt.show()

    return kp1, des1

# fnames : 영상 파이 ㄹ이름 목록
def get_matches(des1, des2, ratio=0.7) :
    # idx1과 idx2에 각 인덱스 저장
    # idx1과 idx2의 같은 위치의 수는 서로 매칭되는 키포인트 인덱스
    idx1, idx2 = [], []
    for i, d1 in enumerate(des1) :
        # 유클리드 거리 계산
        distances = np.linalg.norm(des2 - d1, axis=1) 
        # d1과 가장 가까운 des2에서의 벡터 2개의 위치 산출
        nearest_neighbor = np.argsort(distances)[:2]
        # 가장 가까운 거리 두 개의 비율을 비교
        distance1, distance2 = distances[nearest_neighbor[0]], distances[nearest_neighbor[1]]
        if (distance1/max(1e-6, distance2)) < ratio :
            idx1.append(i)
            idx2.append(nearest_neighbor[0])
            
    return idx1, idx2

def image_feature_match_draw(img1, img2, kp1, kp2, idx1, idx2) :

    # 이미지 붙이기 (이미지 결과 보여주기 준비)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    hdif = int((h2 - h1) / 2)
    
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    newImg = np.zeros((nHeight, nWidth, 3), np.uint8)
    
    for i in range(3) :
        newImg[hdif:hdif + h1, :w1, i] = img1
        newImg[:h2, w1:w1 + w2, i] = img2
    
    # 올바른 매치 올바르지 않은 매치 구하기
    correct, incorrect = 0, 0
    # get_match에서 얻은 정보들로 라인 긋기
    for i in range(len(idx1)) :
        point1 = (int(kp1[idx1[i]].pt[0]), int(kp1[idx1[i]].pt[1] + hdif))
        point2 = (int(kp2[idx2[i]].pt[0] + w1), int(kp2[idx2[i]].pt[1]))
        cv2.line(newImg, point1, point2, (255, 0, 0))
        
        x_delta = int(kp2[idx2[i]].pt[0] + w1) - int(kp1[idx1[i]].pt[0])
        y_delta = int(kp2[idx2[i]].pt[1]) - int(kp1[idx1[i]].pt[1] + hdif)
        gradient = abs(x_delta / y_delta)
        if gradient > 200 :
            incorrect = incorrect + 1
        else :
            correct = correct + 1
        
    plt.imshow(newImg)
    plt.show()
    
def get_matching_feature(img1, img2, NNDR) :
    
    kp1, kp2, _, _, idx1, idx2 = SIFT_feature_matching(img1, img2, NNDR, False)
    number_of_matching = len(idx1)

    matching_keypoint1 = []
    matching_keypoint2 = []

    for index in range(number_of_matching) :                                                    # matching_keypoint1, matching_keypoint2 : 매칭 feature 개수 x 3
        matching_keypoint1.append([kp1[idx1[index]].pt[0], kp1[idx1[index]].pt[1], 1])          # [[x, y, 1],
        matching_keypoint2.append([kp2[idx2[index]].pt[0], kp2[idx2[index]].pt[1], 1])          #  [x, y, 1],
                                                                                                #  [x, y, 1]]
    return matching_keypoint1, matching_keypoint2, number_of_matching


def SIFT_feature_matching(img1, img2, ratio, showImg) :
    kp1, des1 = get_feature_SIFT(img1, showImg)
    kp2, des2 = get_feature_SIFT(img2, showImg)
    idx1, idx2 = get_matches(des1, des2, ratio)
    if showImg :
        image_feature_match_draw(img1, img2, kp1, kp2, idx1, idx2)
    else :
        return kp1, kp2, des1, des2, idx1, idx2