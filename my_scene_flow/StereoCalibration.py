# methods about calibration
# read setereo pic and save calibration info to camera.conf

# import
from ast import List
import json
import pickle
import numpy as np
import cv2
import glob
from matplotlib.font_manager import FontProperties
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import flow_compute
import random
from ff_core.utils import flow_viz

font = FontProperties(
    fname=os.environ['WINDIR'] + '\\Fonts\\kaiu.ttf', size=16)


def get_imgpt(images: List, row=6, col=9, show=False, log_print=False) -> list:
    # 取得照片(像素坐標系)中棋盤方格點(u,v)

    # 設定termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgpoints = []  # 2d points in image plane.
    # print(len(images))
    for fname in images:
        if log_print:
            print("dealing ", fname)
        img = cv2.imread(fname)
        dsize = flow_compute.compute_adaptive_image_size(img.shape[0:2])
        print(f"resize img for chessboard to {dsize}")
        img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if show:
            plt.figure(figsize=(20, 20))
            plt.subplot(121)
            plt.imshow(img[:, :, ::-1])

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (row, col), None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (row, col), corners2, ret)
            if show:
                plt.subplot(122)
                plt.imshow(img[:, :, ::-1])
        else:
            print(f"!!! FIND CHESSBOARD FAIL !!! {fname}")
            imgpoints.append([])
        if show:
            plt.show()
    return imgpoints


class StereoCalibration:
    """立體影像矯正、計算物件

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        __init__ (w=1920, h=1080): 初始化，可設定影像大小，預設為1920*1080
        get_calibration_image_path(dir_l=r"calibration/zed_tv/l", dir_r=r"calibration/zed_tv/r", pic_type=".png", seq_clear=True): 從資料夾中讀取校正用影像，成對影像在資料夾中順序需相同
        compute_stereo_calibration(): 計算校正數據
        load_calibration(path="StereoCalibration"): 讀入校正數據
        save_calibration(path="StereoCalibration"): 導出校正數據
        calibration(dir: str, pic_type=".png", save=True): 從資料夾中讀取校正用影像並計算校正數據
        get_depth(disp, save=False): 利用校正數據計算影象視差結果的深度
        calibration_img_pair(left_img_path: str, right_img_path: str, show=False, save_dir="", ret="path"): 利用校正數據計算影像校正
        __str__():


    Attributes:
        w (int): 影像的寬度
        h (int): 影像的高度
        IMAGE_SIZE (tuples): 影像的大小，等同於(w, h)
        row (int): 校正所需的校正格row數
        col (int): 校正所需的校正格col數
        objectPT (numpy.ndarray): 校正板的世界座標
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F: Essential matrix
        new_size, R1, R2, P1, P2, Q, left_mapx, left_mapy, right_mapx, right_mapy: Stereo rectification
        left_images(list), rignt_images(list): 存有左/右校正用影像的list
    """

    def __init__(self, w=960, h=540):

        # 定義像素大小960*540
        self.w = w
        self.h = h
        self.IMAGE_SIZE = (w, h)

        # 定義相機座標系(X_c, Y_c, Z_c)中的棋盤方格

        self.row = 6
        self.col = 9
        self.objectPt = np.zeros((self.row * self.col, 3), np.float32)
        self.objectPt[:, :2] = np.mgrid[0:self.row,
                                        0:self.col].T.reshape(-1, 2)

        # 定義相機參數
        self.retval = None
        self.cameraMatrix1 = None
        self.distCoeffs1 = None
        self.cameraMatrix2 = None
        self.distCoeffs2 = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None

        self.new_size = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.left_mapx = None
        self.left_mapy = None
        self.right_mapx = None
        self.right_mapy = None

        # 定義校正用影像對
        self.left_images = []
        self.rignt_images = []

    def __str__(self):
        return f"h:{self.h}, w:{self.w}\ncameraMatrix1:\n{self.cameraMatrix1}\ncameraMatrix2:\n{self.cameraMatrix2}"

    def get_all_INFO(self):
        return "Stereo Calibration Info:" + "\nretval" + str(self.retval) + "\ncameraMatrix1" + str(self.cameraMatrix1) + "\ndistCoeffs1" + str(self.distCoeffs1) + "\ncameraMatrix2" + str(self.cameraMatrix2) + "\ndistCoeffs2" + str(self.distCoeffs2) + "\nR" + str(self.R) + "\nT" + str(self.T) + "\nE" + str(self.E) + "\nF" + str(self.F) + "\nR1" + str(self.R1) + "\nR2" + str(self.R2) + "\nP1" + str(self.P1) + "\nP2" + str(self.P2) + "\nQ" + str(self.Q) + "\nleft_mapx" + str(self.left_mapx) + "\nleft_mapy" + str(self.left_mapy) + "\nright_mapx" + str(self.right_mapx) + "\nright_mapy" + str(self.right_mapy)

    def get_calibration_image_path(self, dir_l=r"calibration/zed_tv/l", dir_r=r"calibration/zed_tv/r", pic_type=".png", seq_clear=True):
        self.left_images = [str(f) for f in Path(dir_l).glob("*" + pic_type)]
        self.rignt_images = [str(f) for f in Path(dir_r).glob("*" + pic_type)]
        print(self.left_images)
        print(self.rignt_images)
        pass

    def compute_stereo_calibration(self):
        # 立體影像校正對齊
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 建立校正所需座標
        # 2d points in left image plane.
        limgpts = get_imgpt(self.left_images)
        # 2d points in right image plane.
        rimgpts = get_imgpt(self.rignt_images)

        # 檢查可用性
        i = 0
        while i < len(limgpts):
            if len(limgpts[i]) == 0 or len(rimgpts[i]) == 0:
                del (limgpts[i])
                del (rimgpts[i])
            i += 1

        objpts = [self.objectPt] * len(limgpts)  # 3d point in real world space
        # 計算校正資訊 Essential matrix
        self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpts, limgpts, rimgpts, cameraMatrix1=None, distCoeffs1=None, cameraMatrix2=None, distCoeffs2=None, imageSize=self.IMAGE_SIZE, criteria=criteria, flags=0)

        # 計算 Essential matrix
        scale_factor = 1
        self.new_size = (int(self.w * scale_factor),
                         int(self.h * scale_factor))
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.stereoRectify(
            self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.new_size, self.R, self.T)[0:5]
        self.left_mapx, self.left_mapy = cv2.initUndistortRectifyMap(
            self.cameraMatrix1, self.distCoeffs1, self.R1, self.P1, self.new_size, cv2.CV_32FC1)
        self.right_mapx, self.right_mapy = cv2.initUndistortRectifyMap(
            self.cameraMatrix2, self.distCoeffs2, self.R2, self.P2, self.new_size, cv2.CV_32FC1)

        pass

    def load_calibration(self, path="cameracal"):
        with open(path, 'rb') as f:
            self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F, self.new_size, self.R1, self.R2, self.P1, self.P2, self.Q, self.left_mapx, self.left_mapy, self.right_mapx, self.right_mapy = pickle.load(
                f)
            print("calibration info read")
        pass

    def save_calibration(self, path="cameracal"):
        data2 = [self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F,
                 self.new_size, self.R1, self.R2, self.P1, self.P2, self.Q, self.left_mapx, self.left_mapy, self.right_mapx, self.right_mapy]
        with open(path, 'wb') as f:
            pickle.dump(data2, f)
            print("calibration info saved")
        pass

    def calibration(self, dir: str, pic_type=".png", save=True):
        dir_l = Path(dir) / "l"
        dir_r = Path(dir) / "r"
        self.get_calibration_image_path(
            dir_l=str(dir_l), dir_r=str(dir_r), pic_type=".png")
        self.compute_stereo_calibration()
        if save:
            self.save_calibration()
        pass

    def calibration_img_pair(self, left_img_path: str, right_img_path: str, show=False, save_dir="", ret="path"):
        # 讀取照片
        print("read", left_img_path, right_img_path)
        left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
        # 調整大小
        dsize = flow_compute.compute_adaptive_image_size(left_img.shape[0:2])
        left_img = cv2.resize(left_img, dsize=dsize,
                              interpolation=cv2.INTER_CUBIC)
        right_img = cv2.resize(right_img, dsize=dsize,
                               interpolation=cv2.INTER_CUBIC)
        # 校正
        undistorted_left = cv2.remap(
            left_img, self.left_mapx, self.left_mapy, cv2.INTER_LINEAR)
        undistorted_right = cv2.remap(
            right_img, self.right_mapx, self.right_mapy, cv2.INTER_LINEAR)

        if show:
            print("show calibraion...")
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(np.concatenate(
                [left_img, right_img], axis=1)[:, :, ::-1])
            plt.title('original (同一點不在相同水平線上)', fontproperties=font)
            plt.subplot(1, 2, 2)
            plt.imshow(np.concatenate(
                [undistorted_left, undistorted_right], axis=1)[:, :, ::-1])
            plt.title('stereo rectified (同一點在相同水平線上)', fontproperties=font)
            plt.tight_layout()
            plt.show()

        if save_dir != "":
            cv2.imwrite(
                str(Path(save_dir) / "left" / str(Path(left_img_path).name)), undistorted_left)
            print(
                f"save as {str(Path(save_dir) /'left'/ str(Path(left_img_path).name))}")
            cv2.imwrite(
                str(Path(save_dir) / "right" / str(Path(right_img_path).name)), undistorted_right)
            print(
                f"save as {str(Path(save_dir) /'right'/ str(Path(right_img_path).name))}")

        if ret == "mat":
            return undistorted_left, undistorted_right
        else:
            return str(Path(save_dir) / str(Path(left_img_path).name)), str(Path(save_dir) / str(Path(right_img_path).name))
        pass

    def get_disparity_map_of_img_pair(self, left_img: str, right_img: str, save_dir="", show=False) -> np.array:
        # 準備校正影像
        undistorted_left, undistorted_right = self.calibration_img_pair(
            left_img, right_img, show=show, save_dir=save_dir, ret="mat")
        # print(undistorted_left.shape)

        # 計算flow
        flow = flow_compute.prepare_mat_and_compute_flow(
            undistorted_left, undistorted_right)

        # 將flow結果drop掉flow的y向量
        disparityimg = np.squeeze(np.delete(np.array(flow), 1, axis=2))

        # 回傳shape=(x,y), 代表視差的矩陣
        return disparityimg

    def get_depth_map(self, disp):
        # 透過視差計算三維
        _3dIimg = cv2.reprojectImageTo3D(disp, self.Q)
        # print(_3dIimg)
        return _3dIimg
        pass


def get_flow(img0, img1, show=False):
    if type(img0) == type("str"):
        return flow_compute.prepare_path_and_compute_flow(img0, img1)
    return flow_compute.prepare_mat_and_compute_flow(img0, img1)


def get_scene(img0_depth_map_3d, img1_depth_map_3d, flow):
    print("img0_depth_map_3d:", img0_depth_map_3d.shape)
    print("img1_depth_map_3d:", img1_depth_map_3d.shape)
    print("flow:\n", flow.shape)
    x, y, vec = img0_depth_map_3d.shape
    print("x:", x)
    print("y:", y)
    scene = np.zeros((x, y, 3))
    for ind_x in range(x):
        for ind_y in range(y):
            target_x = int(np.round(ind_x + flow[ind_x][ind_y][0]))
            target_y = int(np.round(ind_y + flow[ind_x][ind_y][1]))
            if 0 <= target_x < x and 0 <= target_y < y:
                # print("ind_x:", ind_x)
                # print("ind_y:", ind_y)
                # print("target_x:", target_x)
                # print("target_y:", target_y)
                # print("img0_depth_map_3d[ind_x][ind_y]:",
                #       img0_depth_map_3d[ind_x][ind_y])
                # print("img1_depth_map_3d[target_x][target_y]:",
                #       img1_depth_map_3d[target_x][target_y])
                scene[ind_x][ind_y] = img1_depth_map_3d[target_x][target_y] - \
                    img0_depth_map_3d[ind_x][ind_y]
        pass
    pass
    return scene


# ===== 視差處理 =====
def plt_disparity_map(disparity_map, title="視差圖", drop=False, drop_min=0, drop_max=1000) -> np.array:
    plt.title(title, fontproperties=font)
    if drop:
        disparity_map = drop_2d_map_bias(
            disparity_map, min=drop_min, max=drop_max)
        plt.imshow
    else:
        plt.imshow(disparity_map, cmap='cividis')
    plt.show()
    return disparity_map


def plt_disparity_map_distribution(disparity_map, title="視差值分布"):
    plt.title(title, fontproperties=font)
    plt.boxplot(disparity_map.flatten())
    plt.show()


def plt_correspondence(left_path, right_path, disparity_map, points_num=15, title="視差對應"):
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    print(disparity_map.shape)
    print(left.shape)
    # 統一調整為成視差圖大小
    left = cv2.resize(left, dsize=(disparity_map.shape[1], disparity_map.shape[0]),
                      interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(right, dsize=(disparity_map.shape[1], disparity_map.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    print(left.shape)
    # 隨機選取點
    rand_points = [(random.randrange(disparity_map.shape[0]), random.randrange(disparity_map.shape[1]))
                   for i in range(points_num)]
    # 準備資料
    left_points = [cv2.KeyPoint(x, y, None) for y, x in rand_points]
    right_points = [cv2.KeyPoint(x + disparity_map[y][x], y, None)
                    for y, x in rand_points]
    matches = [i for i in range(points_num)]
    matches = [cv2.DMatch(i, i, 0) for i in range(points_num)]

    # 檢查資料
    print([i.pt for i in left_points])
    print([i.pt for i in right_points])
    print(left.shape)
    print(right.shape)
    print(disparity_map.shape)

    # 使用cv2.drawMatches功能繪製匹配點
    match_result = cv2.drawMatches(
        left, left_points, right, right_points, matches, None)
    plt.title(title, fontproperties=font)
    plt.imshow(match_result)
    plt.show()
    pass


# ===== 深度處理 =====
def plt_depth_map(depth_map_3d, title="深度圖", drop=False, drop_min=-1000, drop_max=1000):
    if drop:
        plt.title(title + " drop " + str(drop_min) +
                  "~" + str(drop_max), fontproperties=font)
        droped = drop_2d_map_bias(depth_map_3d[:, :, 2])
        plt.imshow(droped, cmap='cividis', vmin=np.min(
            droped), vmax=np.max(droped))
        plt.colorbar()
    else:
        plt.title(title, fontproperties=font)
        plt.imshow(depth_map_3d[:, :, 2], cmap='cividis')
        plt.colorbar()
    plt.show()


def plt_depth_map_3d(depth_map_3d, title="深度圖3D", drop=False, drop_min=-1000, drop_max=1000):
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    if drop:
        x = drop_2d_map_bias(depth_map_3d[:, :, 0]).flatten()
        y = drop_2d_map_bias(depth_map_3d[:, :, 1]).flatten()
        z = drop_2d_map_bias(depth_map_3d[:, :, 2]).flatten()
        plt.title(title + " drop " + str(drop_min) +
                  "~" + str(drop_max), fontproperties=font)
    else:
        x = depth_map_3d[:, :, 0].flatten()
        y = depth_map_3d[:, :, 1].flatten()
        z = depth_map_3d[:, :, 2].flatten()
        plt.title(title, fontproperties=font)
    ax.scatter(x, y, z, c=z)
    plt.show()


def plt_depth_map_distribution(depth_map_3d, title="深度值分布", drop=False, drop_min=-1000, drop_max=1000):
    if drop:
        x = drop_2d_map_bias(depth_map_3d[:, :, 0]).flatten()
        y = drop_2d_map_bias(depth_map_3d[:, :, 1]).flatten()
        z = drop_2d_map_bias(depth_map_3d[:, :, 2]).flatten()
        plt.title(title + " drop " + str(drop_min) +
                  "~" + str(drop_max), fontproperties=font)
    else:
        x = depth_map_3d[:, :, 0].flatten()
        y = depth_map_3d[:, :, 1].flatten()
        z = depth_map_3d[:, :, 2].flatten()
        plt.title(title, fontproperties=font)
    plt.boxplot([x, y, z])
    plt.show()


# ===== 場景流處理 =====
def plt_scene_flow_distribution(scene_flow, title="場景流值分布"):
    x = scene_flow[:, :, 0].flatten()
    y = scene_flow[:, :, 1].flatten()
    z = scene_flow[:, :, 2].flatten()
    plt.title(title, fontproperties=font)
    plt.boxplot([x, y, z])
    plt.show()
    pass


def plt_scene_flow(scene_flow, title="場景流", vmin=None, vmax=None):
    """plt scene flow in different view

    Args:
        scene_flow (3d_vector): input scene flow in a matrix of 3d vector in 540*960*3 resolution
        title (str, optional): 圖表標題. Defaults to "場景流".
    """
    # build plt
    plt.title(title, fontproperties=font)
    # x軸
    plt.subplot(3, 1, 1)
    plt.imshow(scene_flow[:, :, 0], vmin=vmin, vmax=vmax, cmap='cividis')
    plt.colorbar()
    # y軸
    plt.subplot(3, 1, 2)
    plt.imshow(scene_flow[:, :, 1], vmin=vmin, vmax=vmax, cmap='cividis')
    plt.colorbar()
    # z軸
    plt.subplot(3, 1, 3)
    plt.imshow(scene_flow[:, :, 2], vmin=vmin, vmax=vmax, cmap='cividis')
    plt.colorbar()
    plt.show()
    pass

# ===== 光流處理 =====


def plt_flow(flow, title="光流"):
    flow_img = flow_viz.flow_to_image(flow)
    plt.title(title, fontproperties=font)
    plt.imshow(flow_img)
    plt.colorbar()
    plt.show()


def drop_2d_map_bias(depth_map_2d, min=-1000, max=1000, drop_as=0):
    depth_map_2d[(depth_map_2d < min) | (depth_map_2d > max)] = drop_as
    return depth_map_2d


def test_calibration(show=False):
    """test StereoCalibration，從資料夾讀取校正照片並進行校正，最後儲存校正資料
    """
    my_stereo_calibration = StereoCalibration()
    my_stereo_calibration.calibration(
        r"calibration/zed_tv", pic_type=".png", save=True)
    # print(my_stereo_calibration)
    l_img, r_img = my_stereo_calibration.calibration_img_pair(
        r"calibration/zed_tv/l/Explorer_HD1080_SN30323411_10-59-39-l.png", r"calibration/zed_tv/r/Explorer_HD1080_SN30323411_10-59-39-r.png", save_dir=r"calibration/resault/zed_tv", ret='mat')
    print(l_img.shape, r_img.shape)
    if show:
        plt.imshow(np.concatenate([l_img, r_img], axis=1)[:, :, ::-1])
        plt.title('stereo rectified 測試校正後影像(by test_calibration())',
                  fontproperties=font)
        plt.show()
    del (my_stereo_calibration)
    print("===== test_calibration end =====")
    pass


def test_depth(left_img=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_l\2022-10-13-16-35-07_l.mp4_frame_0001.jpg", right_img=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_r\2022-10-13-16-35-07_r.mp4_frame_0001.jpg"):
    """載入校正資料並計算視差、深度"""
    my_stereo_calibration = StereoCalibration()
    my_stereo_calibration.load_calibration(path="cameracal")
    # print(my_stereo_calibration)
    print("===== count disparity =====")
    disparity_map = my_stereo_calibration.get_disparity_map_of_img_pair(
        left_img=left_img, right_img=right_img, show=False)
    print("disparity_map:\n", type(disparity_map), '\n',
          disparity_map.shape, '\n', disparity_map)
    print(disparity_map.min(), disparity_map.max())
    print("===== call disparity plt =====")
    plt_disparity_map(disparity_map)
    plt_disparity_map_distribution(disparity_map)

    print("===== call correspondence plt =====")
    plt_correspondence(left_img, right_img, disparity_map)
    print("===== count depth =====")
    depth_map = my_stereo_calibration.get_depth_map(disparity_map)
    print("depth_map:\n", depth_map.shape, '\n', depth_map)

    print("===== call depth_map plt =====")
    plt_depth_map(depth_map)
    # plt_depth_map_3d(depth_map)
    plt_depth_map_distribution(depth_map)

    print("===== drop depthmap bias =====")
    # plt_depth_map(depth_map, drop=True, drop_min=-1000, drop_max=1000)
    # plt_depth_map(depth_map, drop=True, drop_min=-500, drop_max=500)
    # plt_depth_map(depth_map, drop=True, drop_min=-250, drop_max=250)
    # plt_depth_map_3d(depth_map, drop=True)
    # plt_depth_map_distribution(depth_map, drop=True)

    print("===== test_depth end =====")
    pass


def test_scene_flow(t0_l=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_l\2022-10-13-16-35-07_l.mp4_frame_0001.jpg", t0_r=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_r\2022-10-13-16-35-07_r.mp4_frame_0001.jpg", t1_l=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_l\2022-10-13-16-35-07_l.mp4_frame_0002.jpg", t1_r=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_r\2022-10-13-16-35-07_r.mp4_frame_0002.jpg"):
    # 載入校正資料
    my_stereo_calibration = StereoCalibration()
    my_stereo_calibration.load_calibration(path="cameracal")
    # 計算t0深度
    print("===== prepare t0 depth map =====")
    t0_disparity_map = my_stereo_calibration.get_disparity_map_of_img_pair(
        left_img=t0_l, right_img=t0_r, show=False)
    t0_depth_map = my_stereo_calibration.get_depth_map(t0_disparity_map)
    # 計算t1深度
    print("===== prepare t1 depth map =====")
    t1_disparity_map = my_stereo_calibration.get_disparity_map_of_img_pair(
        left_img=t1_l, right_img=t1_r, show=False)
    t1_depth_map = my_stereo_calibration.get_depth_map(t1_disparity_map)
    print("===== call depth_map plt =====")
    # plt_depth_map(t0_depth_map)
    # plt_depth_map(t1_depth_map)
    # 計算t0>t1光流
    print("===== prepare flow map =====")
    flow = get_flow(t0_l, t1_l)
    print("flow_map:\n", flow.shape, '\n', flow)
    # 計算光流變化+深度變化
    print("===== get scene flow =====")
    scene = get_scene(t0_depth_map, t1_depth_map, flow)
    print("scene_map:\n", scene.shape, '\n', scene)
    plt_scene_flow_distribution(scene)
    plt_scene_flow(scene)
    return scene, flow, t0_depth_map, t1_depth_map
    pass


def test_zed_55():
    # 第一組
    # 計算場景流
    scene_01, flow_01, t0_depth_map, t1_depth_map = test_scene_flow(r"datasets\zed_55\resize\01_01_l.png",
                                                                    r"datasets\zed_55\resize\01_01_r.png", r"datasets\zed_55\resize\01_02_l.png", r"datasets\zed_55\resize\01_02_r.png")
    # plt
    plt_depth_map(t0_depth_map)
    plt_depth_map(t1_depth_map)
    plt_flow(flow_01)
    # 在目標上取5個點
    points = [(337, 453), (388, 480), (353, 511), (64, 435), (847, 59)]
    # 取5個點的值
    for point in points:
        # scene flow value
        print(f"===== point {point} =====")
        print(f"scene flow: {scene_01[point[1]][point[0]]}")
        # 這邊應該還要寫出前後的位置比較
        print(f"flow: {flow_01[point[1]][point[0]]}")
    print(type(flow_01))
    pass


def test_print_cal_info():
    # 載入校正資料
    my_stereo_calibration = StereoCalibration()
    my_stereo_calibration.load_calibration(path="cameracal")
    print(my_stereo_calibration.get_all_INFO())


if __name__ == '__main__':
    # test_calibration()
    # test_print_cal_info()
    # test_depth(left_img=r"calibration\zed_tv\l\Explorer_HD1080_SN30323411_10-59-39-l.png",
    #            right_img=r"calibration\zed_tv\r\Explorer_HD1080_SN30323411_10-59-39-r.png")
    # test_depth(left_img=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_l\2022-10-13-16-35-07_l.mp4_frame_0001.jpg",
    #            right_img=r"datasets\zed_water\20221011_A13\2022-10-13-16-35-07\2022-10-13-16-35-07_r\2022-10-13-16-35-07_r.mp4_frame_0001.jpg")
    # test_scene_flow()
    test_zed_55()
    pass
