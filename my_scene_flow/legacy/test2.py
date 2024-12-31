import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

left_path = r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_l\2022-10-13-16-35-07_l.mp4_frame_0001.jpg"
right_path = r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_r\2022-10-13-16-35-07_r.mp4_frame_0001.jpg"

main_size = (540, 960)
disparity_map = np.ones(main_size)
points_num = 15

left = cv2.imread(left_path, cv2.IMREAD_COLOR)
right = cv2.imread(right_path, cv2.IMREAD_COLOR)
print(left.shape)

left = cv2.resize(left, dsize=(disparity_map.shape[1], disparity_map.shape[0]),
                  interpolation=cv2.INTER_CUBIC)
right = cv2.resize(right, dsize=(disparity_map.shape[1], disparity_map.shape[0]),
                   interpolation=cv2.INTER_CUBIC)
print(left.shape)

# 隨機選取點
rand_points = [(random.randrange(disparity_map.shape[0]), random.randrange(disparity_map.shape[1]))
               for i in range(points_num)]
print(rand_points)

# 準備資料
left_points = [cv2.KeyPoint(x, y, None) for y, x in rand_points]
right_points = [cv2.KeyPoint(x + disparity_map[y][x], y, None)
                for y, x in rand_points]
matches = [i for i in range(points_num)]
matches = [cv2.DMatch(i, i, 0) for i in range(points_num)]

# 檢查資料
print([i.pt for i in left_points])
print([i.pt for i in right_points])

# 使用cv2.drawMatches功能繪製匹配點
match_result = cv2.drawMatches(
    left, left_points, right, right_points, matches, None)
plt.imshow(match_result)
plt.show()


def plt_correspondence(left_path, right_path, disparity_map, points_num=15):
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    # 統一調整為成視差圖大小
    left = cv2.resize(left, dsize=disparity_map.shape,
                      interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(right, dsize=disparity_map.shape,
                       interpolation=cv2.INTER_CUBIC)
    # 隨機選取點
    rand_points = [(random.randrange(disparity_map.shape[0]), random.randrange(disparity_map.shape[1]))
                   for i in range(points_num)]
    # 準備資料
    left_points = [cv2.KeyPoint(x, y,)]
    right_points = [(y, x + disparity_map[y][x]) for y, x in left_points]
    matches = [i for i in range(points_num)]
    # 使用cv2.drawMatches功能繪製匹配點
    match_result = cv2.drawMatches(
        left, left_points, right, right_points, matches, None)
    plt.imshow(match_result)
    plt.show()
    pass
