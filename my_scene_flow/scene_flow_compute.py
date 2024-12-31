# methods about scene flow
# include scene flow, depth

# import
from ast import List
import json
import pickle
import numpy as np
import cv2
import glob
import os
from pathlib import Path
from PIL import Image
# import matplotlib
import matplotlib.pyplot as plt
from sympy import Q, Array
# from ff_depth.ff_depth import depth_read
import flow_compute
import random
from ff_core.utils import flow_viz

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus'] = False

MAX_SCENE_FLOW = 10

# ===== 校正 =====
# drop2d矩陣極端值


def drop_2d_map_bias(depth_map_2d: np.ndarray, min=-1000, max=1000):
    depth_map_2d[(depth_map_2d < min)] = min
    depth_map_2d[(depth_map_2d > max)] = max
    return depth_map_2d


# 解算P矩陣
def decompose_projection_matrix(P):
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T / T[3]
    return K, R, T


# 計算Q矩陣
def decompute_Q_matrix(P1, P2):
    # reference: https://stackoverflow.com/a/28317841/17732660

    f = P1[0][0]
    cx1 = P1[0][2]
    cy = P1[1][2]
    cx2 = P2[0][2]
    Tx = P2[0][3] / f

    Q = np.array([[1, 0, 0, -cx1], [0, 1, 0, -cy], [0, 0, 0, f],
                 [0, 0, -1 / Tx, (cx1 - cx2) / Tx]])
    return Q
    pass


def get_ZED_Q_matrix() -> np.ndarray:
    Q = np.array([[1., 0., 0., -697.43959045],
                  [0., 1., 0., -236.69963646],
                  [0., 0., 0., 929.4452],
                  [0., 0., -16.77215164, 0.]])
    return Q


# 取得KITTI Q矩陣
def get_KITTI_Q_matrix(name: str) -> np.ndarray:
    KITTI_path = Path(
        r'E:\datasets\KITTI_sceneflow\calibration_files\training\calib_cam_to_cam')
    calib_path = KITTI_path / (name + ".txt")
    calib = calib_path.read_text()
    # with calib_path.open('r') as f:
    #     calib = f.readlines()
    calib = calib.split('\n')
    # print(len(calib))
    # print(calib)

    K_02 = np.array(calib[19].strip().split()[1:]
                    ).astype(float).reshape((3, 3))
    D_02 = np.array(calib[20].strip().split()[1:]
                    ).astype(float).flatten()
    R_02 = np.array(calib[21].strip().split()[1:]
                    ).astype(float).reshape((3, 3))
    T_02 = np.array(calib[22].strip().split()[1:]
                    ).astype(float).flatten()
    S_rect_02 = np.array(calib[23].strip().split()[
                         1:]).astype(float).flatten()
    R_rect_02 = np.array(calib[24].strip().split()[
                         1:]).astype(float).reshape((3, 3))
    P_rect_02 = np.array(calib[25].strip().split()[
                         1:]).astype(float).reshape((3, 4))
    K_03 = np.array(calib[27].strip().split()[1:]
                    ).astype(float).reshape((3, 3))
    D_03 = np.array(calib[28].strip().split()[1:]
                    ).astype(float).flatten()
    R_03 = np.array(calib[29].strip().split()[1:]
                    ).astype(float).reshape((3, 3))
    R_rect_03 = np.array(calib[32].strip().split()[
                         1:]).astype(float).reshape((3, 3))
    P_rect_03 = np.array(calib[33].strip().split(
    )[1:]).astype(float).reshape((3, 4))

    R1, R2, P1, P2, Q, RECT1, RECT2 = cv2.stereoRectify(
        K_02, D_02, K_03, D_03, S_rect_02.astype(int), R_rect_02, T_02, R_rect_03)
    P_rect_02_err = sum(abs(P1.flatten() - P_rect_02.flatten())) / 12
    P_rect_03_err = sum(abs(P2.flatten() - P_rect_03.flatten())) / 12
    # print(f"P_rect_02:\n{P_rect_02}")
    # print(f"P1:\n{P1}")
    # print(f"P_rect_02_err: {P_rect_02_err}")
    # print(f"P_rect_03:\n{P_rect_03}")
    # print(f"P2:\n{P2}")
    # print(f"P_rect_03_err: {P_rect_03_err}")
    return Q


# ===== 視差處理 =====
# 計算視差
def get_disparity(left_img: np.ndarray, right_img: np.ndarray, viz=False, title=None):
    # 取得左右影像flow
    flow = flow_compute.prepare_mat_and_compute_flow(
        left_img, right_img)
    if viz:
        # for draw
        plt_flow(flow, title=title)
    # 將flow結果drop掉flow的y向量
    disparityimg = np.squeeze(np.delete(np.array(flow), 1, axis=2))
    # 回傳shape=(x,y), 代表視差的矩陣
    return disparityimg


# 繪製視差
def plt_disparity_map(disparity_map: np.ndarray, title="視差圖", disp_max=100):
    plt.title(title)
    plt.imshow(disparity_map, cmap='cividis', vmax=disp_max)
    plt.colorbar()
    plt.show()


# 繪製KITTI視差
def plt_KITTI_disparity_map(disparity_path: str, title="視差圖", disp_max=100):
    disp = cv2.imread(str(disparity_path), cv2.IMREAD_UNCHANGED)
    disp = disp.astype(float) / 256.0
    plt.title(title)
    plt.imshow(disp, cmap='cividis', vmax=disp_max)
    plt.colorbar()
    plt.show()


# 繪製視差分布值
def plt_disparity_map_distribution(disparity_map: np.ndarray, title="視差值分布"):
    plt.title(title)
    plt.boxplot(disparity_map.flatten())
    plt.show()


# 繪製左右影像視差對應點
def plt_correspondence(left_path: str, right_path: str, disparity_map: np.ndarray, points_num=15, title="視差對應"):
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
    plt.title(title)
    plt.imshow(match_result)
    plt.show()
    pass


# ===== 深度處理 =====
# 計算各pixel世界座標位置
def reproject_depth(left: np.ndarray, right: np.ndarray, Q: np.ndarray) -> np.ndarray:
    disp = get_disparity(left, right)
    _3d_map = cv2.reprojectImageTo3D(disp, Q)
    return _3d_map


def reproject_depth_from_disp(disp: np.ndarray, Q: np.ndarray) -> np.ndarray:
    _3d_map = cv2.reprojectImageTo3D(disp, Q)
    return _3d_map


# drop深度圖
def drop_depth_map(depth_map: np.ndarray):
    x = drop_2d_map_bias(depth_map[:, :, 0])
    y = drop_2d_map_bias(depth_map[:, :, 1])
    z = drop_2d_map_bias(depth_map[:, :, 2])
    return x, y, z


# 繪製深度圖
def plt_depth_map(depth_map: np.ndarray, title="深度圖", depth_max=100):
    plt.title(title)
    plt.imshow(depth_map[:, :, 2], cmap='cividis')
    # plt.colorbar()
    plt.show()


# 繪製KITTI深度圖
def plt_KITTI_depth_map(depth_path: str, title="深度圖", depth_max=100):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(float) / 256.0
    plt.title(title)
    plt.imshow(depth, cmap='cividis', vmax=depth_max)
    # plt.colorbar()
    plt.show()


# 繪製3D深度圖
def plt_3d_depth_map(depth_map: np.ndarray, title="深度圖3D"):
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    x = depth_map[:, :, 0].flatten()
    y = depth_map[:, :, 1].flatten()
    z = depth_map[:, :, 2].flatten()
    plt.title(title)
    ax.scatter(x, y, z, c=z)
    plt.show()


# 繪製深度圖分布
def plt_depth_map_distribution(depth_map: np.ndarray, title="深度值分布"):
    x = depth_map[:, :, 0].flatten()
    y = depth_map[:, :, 1].flatten()
    z = depth_map[:, :, 2].flatten()
    plt.title(title)
    plt.boxplot([x, y, z])
    plt.show()


def plt_depth_from_top(depth_map: np.ndarray, title="深度俯視圖"):

    pass


# ===== 光流處理 optical flow =====
# 計算光流
def get_flow(img0: np.ndarray, img1: np.ndarray):
    if type(img0) == type("str"):
        return flow_compute.prepare_path_and_compute_flow(img0, img1)
    return flow_compute.prepare_mat_and_compute_flow(img0, img1)


def plt_flow(flow, title="光流"):
    flow_img = flow_viz.flow_to_image(flow)
    plt.title(title)
    plt.imshow(flow_img)
    plt.show()


# 繪製KITTI光流圖
def plt_KITTI_flow(flow_path: str, title="深度圖"):
    """
    KITTI:
    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
    the u-/v-flow into floating point values, convert the value to float, subtract
    2^15 and divide the result by 64.0:

    flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
    flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
    valid(u,v)  = (bool)I(u,v,3);
    """
    flow = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
    # print(type(flow))
    # print(flow[250, 250])
    flow = (flow.astype(float) - 2 ** 15) / 64.0
    # 重新調整channel，因為cv2.imread是讀取GBR，我們要RGB的[0,1]
    flow = flow[:, :, (2, 1)]
    # print(flow[250, 250])
    flow_img = flow_viz.flow_to_image(flow)
    plt.title(title)
    plt.imshow(flow_img)
    plt.show()


# ===== 場景流處理 scene flow =====
def get_scene(img0_3d, img1_3d, flow):
    print(flow[190][810])
    # print("img0_depth_map_3d:", img0_3d.shape)
    # print("img1_depth_map_3d:", img1_3d.shape)
    # print("flow:\n", flow.shape)
    y, x, vec = img0_3d.shape
    print("x:", x)
    print("y:", y)
    scene = np.zeros((y, x, 3))
    for ind_x in range(x):
        for ind_y in range(y):
            target_x = int(np.round(ind_x + flow[ind_y][ind_x][0]))
            target_y = int(np.round(ind_y + flow[ind_y][ind_x][1]))
            if 0 <= target_x < x and 0 <= target_y < y:
                # print("ind_x:", ind_x)
                # print("ind_y:", ind_y)
                # print("target_x:", target_x)
                # print("target_y:", target_y)
                # print("img0_depth_map_3d[ind_x][ind_y]:",
                #       img0_3d[ind_x][ind_y])
                # print("img1_depth_map_3d[target_x][target_y]:",
                #       img1_3d[target_x][target_y])
                if not np.isinf([img1_3d[target_y][target_x], img0_3d[ind_y][ind_x]]).any():
                    temp = img1_3d[target_y][target_x] - img0_3d[ind_y][ind_x]
                    scene[ind_y][ind_x] = temp
    print(scene)
    return scene


# 繪製場景流分布
def plt_scene_flow_distribution(scene_flow, title="場景流值分布"):
    x = scene_flow[:, :, 0].flatten()
    y = scene_flow[:, :, 1].flatten()
    z = scene_flow[:, :, 2].flatten()
    plt.title(title)
    plt.boxplot([x, y, z])
    plt.show()
    pass


# 繪製場景流
def plt_scene_flow(scene_flow, title="場景流", vmin=None, vmax=None):
    """plt scene flow in different view

    Args:
        scene_flow (3d_vector): input scene flow in a matrix of 3d vector in 540*960*3 resolution
        title (str, optional): 圖表標題. Defaults to "場景流".
    """
    import matplotlib.colors as colors
    # build plt
    fig, axarr = plt.subplots(3)
    fig.suptitle(title)
    # x軸
    vmin = scene_flow[:, :, 0].min()
    vmax = scene_flow[:, :, 0].max()
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = axarr[0].imshow(scene_flow[:, :, 0], cmap="bwr", norm=norm)
    plt.colorbar(im, ax=axarr[0])
    axarr[0].set_title('x 位移(正向上)')
    # y軸
    vmin = scene_flow[:, :, 1].min()
    vmax = scene_flow[:, :, 1].max()
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = axarr[1].imshow(scene_flow[:, :, 1], cmap="bwr", norm=norm)
    plt.colorbar(im, ax=axarr[1])
    axarr[1].set_title('y 位移(正向右)')
    # z軸
    vmin = scene_flow[:, :, 2].min()
    vmax = scene_flow[:, :, 2].max()
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = axarr[2].imshow(scene_flow[:, :, 2], cmap="bwr", norm=norm)
    plt.colorbar(im, ax=axarr[2])
    axarr[2].set_title('z 位移(正向前)')
    plt.show()
    pass


def plt_KITTI_scene_flow(id="000010", title="KITTI scene"):

    # get KITTI Q
    Q = get_KITTI_Q_matrix(id)

    # get disp
    disparity_path_1 = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\disp_occ_0\000010_10.png"
    disparity_path_2 = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\disp_occ_1\000010_10.png"
    disp1 = cv2.imread(str(disparity_path_1), cv2.IMREAD_UNCHANGED)
    disp1 = disp1.astype(float) / 256.0
    disp1 = disp1.astype(np.float32)
    disp2 = cv2.imread(str(disparity_path_2), cv2.IMREAD_UNCHANGED)
    disp2 = disp2.astype(float) / 256.0
    disp2 = disp2.astype(np.float32)
    print(type(disp1[0][0]))
    print(disp1.shape)

    # get depth
    print(disp1)
    print(disp1.max())
    _3d_map_1 = cv2.reprojectImageTo3D(disp1, Q)
    _3d_map_2 = cv2.reprojectImageTo3D(disp2, Q)

    # get flow
    flow_path = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\flow_occ\000010_10.png"
    flow = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
    flow = (flow.astype(float) - 2 ** 15) / 64.0
    # 重新調整channel，因為cv2.imread是讀取GBR，我們要RGB的[0,1]
    flow = flow[:, :, (2, 1)]
    plt_flow(flow)

    # get scene
    scene = get_scene(_3d_map_1, _3d_map_2, flow)
    print(scene.shape)
    # drop = scene.max(-1) > 10
    # scene[drop] = [0, 0, 0]
    plt_scene_flow(scene, "KITTI 000010 場景流基準值")


def plt_KITTI_scene_flow_o3d(id="000010", title="KITTI scene"):
    # get rgb
    KITTI_training_path = Path(
        r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training")
    image_2_path = KITTI_training_path / "image_2" / (id + "_10.png")
    rgb = cv2.cvtColor(cv2.imread(str(image_2_path)), cv2.COLOR_BGR2RGB)

    # get KITTI Q
    Q = get_KITTI_Q_matrix(id)

    # get disp
    disparity_path_1 = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\disp_occ_0\000010_10.png"
    disparity_path_2 = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\disp_occ_1\000010_10.png"
    disp1 = cv2.imread(str(disparity_path_1), cv2.IMREAD_UNCHANGED)
    disp1 = disp1.astype(float) / 256.0
    disp1 = disp1.astype(np.float32)
    disp2 = cv2.imread(str(disparity_path_2), cv2.IMREAD_UNCHANGED)
    disp2 = disp2.astype(float) / 256.0
    disp2 = disp2.astype(np.float32)
    print(type(disp1[0][0]))
    print(disp1.shape)

    # get depth
    print(disp1)
    print(disp1.max())
    _3d_map_1 = cv2.reprojectImageTo3D(disp1, Q)
    _3d_map_2 = cv2.reprojectImageTo3D(disp2, Q)

    # get flow
    flow_path = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\flow_occ\000010_10.png"
    flow = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
    flow = (flow.astype(float) - 2 ** 15) / 64.0
    # 重新調整channel，因為cv2.imread是讀取GBR，我們要RGB的[0,1]
    flow = flow[:, :, (2, 1)]

    # get scene
    scene = get_scene(_3d_map_1, _3d_map_2, flow)

    # o3d plt
    plt_o3d_scene_flow(_3d_map_1, rgb, scene, title=title)

    pass


def plt_o3d_scene_flow(xyz, pic, scene, title="scene flow o3d"):
    import open3d as o3d
    xyz = xyz.reshape(-1, 3)
    pic = pic.reshape(-1, 3)
    print(xyz.shape)
    print(xyz)
    print(pic.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(pic)
    print(pcd.get_center())

    o3d.visualization.draw_geometries([pcd], window_name="test")

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(height=480, width=640)
    # vis.add_geometry(pcd)
    # vis.update_renderer()
    # keep_running = True
    # while keep_running:
    #     keep_running = vis.poll_events()
    #     vis.update_renderer()
    # vis.destroy_window()

    # for x, y, z, u, v, w in zip(xyz, scene):
    #     pass
    pass

    # https://github.com/visinf/self-mono-sf/blob/master/demo/demo_generator/run.py
    # pts1_color = np.reshape(flow_img, (hh * ww, 3))
    # pts1_color = np.concatenate((pts1_color, bb_colors), axis=0)

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pts1_np)
    # pcd1.colors = o3d.utility.Vector3dVector(pts1_color)

    # bbox = o3d.geometry.AxisAlignedBoundingBox(min_crop, max_crop)
    # pcd1 = pcd1.crop(bbox)


def get_color(u, v):
    r = 0
    g = 0
    b = 0
    return r, g, b


# 把場景流視覺化。分為xy平面(後視圖)、yz平面(側視圖)、zx平面(上視圖)
def plt_visualize_scene_flow(scene_flow, title="場景流視覺化"):

    x_flow = scene_flow[:, :, 0]
    y_flow = scene_flow[:, :, 1]
    z_flow = scene_flow[:, :, 2]
    plt.title("xy平面(後視圖)")
    xy_flow = np.dstack((x_flow, y_flow))
    print(xy_flow)

    plt.title("yz平面(側視圖)")
    yz_flow = np.dstack((y_flow, z_flow))
    print(yz_flow)
    plt.title("zx平面(上視圖)")
    zx_flow = np.dstack((z_flow, x_flow))
    print(zx_flow)
    pass


# for KITTI
def test_KITTI(id="000010"):
    #
    KITTI_training_path = Path(
        r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training")
    t0_l_path = KITTI_training_path / "image_2" / (id + "_10.png")
    t0_r_path = KITTI_training_path / "image_3" / (id + "_10.png")
    t1_l_path = KITTI_training_path / "image_2" / (id + "_11.png")
    t1_r_path = KITTI_training_path / "image_3" / (id + "_11.png")
    # print(t0_l_path)
    # print(str(t0_l_path))
    t0_l = cv2.cvtColor(cv2.imread(str(t0_l_path)), cv2.COLOR_BGR2RGB)
    t0_r = cv2.cvtColor(cv2.imread(str(t0_r_path)), cv2.COLOR_BGR2RGB)
    t1_l = cv2.cvtColor(cv2.imread(str(t1_l_path)), cv2.COLOR_BGR2RGB)
    t1_r = cv2.cvtColor(cv2.imread(str(t1_r_path)), cv2.COLOR_BGR2RGB)
    # t0_disparity_map = get_disparity(
    #     left_img=t0_l, right_img=t0_r, viz=True, title="000010 t0 左右影像光流")
    # t1_disparity_map = get_disparity(
    #     left_img=t1_l, right_img=t1_r, viz=True, title="000010 t1 左右影像光流")
    # plt_disparity_map(np.abs(t0_disparity_map), id + " t0 視差圖")
    # plt_disparity_map(t1_disparity_map, id + " t1 視差圖")
    Q = get_KITTI_Q_matrix(id)

    # ===== t0深度圖 =====
    _3d_t0 = reproject_depth(t0_l, t0_r, Q)
    # plt_depth_map(_3d_t0, title=id + " t0 深度圖")
    # plt_depth_map_distribution(_3d_t0)
    # ===== t0深度圖 drop =====
    # _3d_t0[(_3d_t0 > 100)] = 100
    # plt_depth_map(_3d_t0, title=id + " t0 深度圖 drop 100")
    # plt_depth_map_distribution(_3d_t0)
    # _3d_t0[(_3d_t0 > 50)] = 50
    # plt_depth_map(_3d_t0, title=id + " t0 深度圖")
    # plt_depth_map_distribution(_3d_t0)
    # _3d_t0[(_3d_t0 > 20)] = 20
    # plt_depth_map(_3d_t0, title=id + " t0 深度圖 drop 20")
    # plt_depth_map_distribution(_3d_t0)
    # _3d_t0[(_3d_t0 > 10)] = 10
    # plt_depth_map(_3d_t0, title=id + " t0 深度圖")
    # plt_depth_map_distribution(_3d_t0)
    # plt_3d_depth_map(_3d_t0, title=id + "t0 深度圖3D")

    # ===== t1深度圖 =====
    _3d_t1 = reproject_depth(t1_l, t1_r, Q)
    # plt_depth_map(_3d_t1, title=id + " t1 深度圖")
    # plt_3d_depth_map(_3d_t1, title=id + "t1 深度圖3D")

    # ===== 光流 =====
    flow = get_flow(t0_l, t1_l)
    plt_flow(flow, title="000010 t0 t1 光流")

    # ===== 場景流 =====
    scene = get_scene(_3d_t0, _3d_t1, flow)
    plt_visualize_scene_flow(scene)
    plt_scene_flow(scene)
    plt_o3d_scene_flow(scene)
    return  # for draw
    # plt_scene_flow_distribution(scene)
    # ===== 場景流 drop =====
    # scene[(scene > 100)] = 100
    # scene[(scene < -100)] = -100
    # plt_scene_flow_distribution(scene, title=id + "場景流值分布 drop 100")
    # plt_scene_flow(scene, title=id + "場景流值分布 drop 100")
    # scene[(scene > 50)] = 50
    # scene[(scene < -50)] = -50
    # plt_scene_flow_distribution(scene, title=id + "場景流值分布 drop 50")
    # plt_scene_flow(scene, title=id + "場景流值分布 drop 50")
    # scene[(scene > 20)] = 20
    # scene[(scene < -20)] = -20
    # plt_scene_flow_distribution(scene, title=id + "場景流值分布 drop 20")
    # plt_scene_flow(scene, title=id + "場景流值分布 drop 20")
    # scene[(scene > 10)] = 10
    # scene[(scene < -10)] = -10
    # plt_scene_flow_distribution(scene, title=id + "場景流值分布 drop 10")
    # plt_scene_flow(scene, title=id + "場景流值分布 drop 10")
    pass


# for master_dataset
def test_A13(date="a13_2022-10-28-07-00-04", t0="00000000", t1="00000001"):
    master_dataset_path = Path(r"E:\datasets\ck_master_dataset")
    t0_l_path = master_dataset_path / "png_960" / date / "r" / (t0 + ".png")
    t0_r_path = master_dataset_path / "png_960" / date / "l" / (t0 + ".png")
    t1_l_path = master_dataset_path / "png_960" / date / "r" / (t1 + ".png")
    t1_r_path = master_dataset_path / "png_960" / date / "l" / (t1 + ".png")
    t0_l = cv2.cvtColor(cv2.imread(str(t0_l_path)), cv2.COLOR_BGR2RGB)
    t0_r = cv2.cvtColor(cv2.imread(str(t0_r_path)), cv2.COLOR_BGR2RGB)
    t1_l = cv2.cvtColor(cv2.imread(str(t1_l_path)), cv2.COLOR_BGR2RGB)
    t1_r = cv2.cvtColor(cv2.imread(str(t1_r_path)), cv2.COLOR_BGR2RGB)
    # t0_disparity_map = get_disparity(
    #     left_img=t0_l, right_img=t0_r, viz=False, title="00000000 t0 左右影像光流")
    # t1_disparity_map = get_disparity(
    #     left_img=t1_l, right_img=t1_r, viz=True, title="00000001 t1 左右影像光流")
    # plt_disparity_map(np.abs(t0_disparity_map), t0 + " 視差圖")
    Q = get_ZED_Q_matrix()

    # ===== t0深度圖 =====
    _3d_t0 = reproject_depth(t0_l, t0_r, Q)
    # ===== t1深度圖 =====
    _3d_t1 = reproject_depth(t1_l, t1_r, Q)
    # ===== 光流 =====
    flow = get_flow(t0_l, t1_l)
    # plt_flow(flow, title="000010 t0 t1 光流")
    # ===== 場景流 =====
    scene = get_scene(_3d_t0, _3d_t1, flow)
    # plt_visualize_scene_flow(scene)
    plt_scene_flow(scene)


def draw_kitti_disp():
    path = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\disp_occ_0\000010_10.png"
    title = "000010 t0 視差圖基準值"
    plt_KITTI_disparity_map(path, title)


def draw_kitti_depth():
    path = r"E:\datasets\KITTI_Depth Prediction\data_depth_annotated\train\2011_09_26_drive_0009_sync\proj_depth\groundtruth\image_02\0000000384.png"
    title = "000010 t0 深度圖基準值"
    plt_KITTI_depth_map(path, title, 60)


def draw_kitti_flow():
    path = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\flow_occ\000010_10.png"
    title = "000010 t0 t1 光流基準值"
    plt_KITTI_flow(path, title)


def draw_kitti_scene():
    path = r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training\flow_occ\000010_10.png"
    title = "000010 場景流基準值"
    plt_KITTI_scene_flow("000010", title)
    # plt_KITTI_scene_flow_o3d("000010", title)


def count_KITTI_dissp_err(disp: np.array, disp_gt_path: str, show=False) -> tuple[float, np.array]:
    # read gt
    disp_gt = cv2.imread(str(disp_gt_path), cv2.IMREAD_UNCHANGED)
    disp_gt = disp_gt.astype(float) / 256.0
    disp = abs(disp)
    print(f"disp: {disp.min()} <-----> {disp.max()}")
    print(f"disp_gt: {disp_gt.min()} <-----> {disp_gt.max()}")
    print(f"original disp: {disp.shape}, disp_gt: {disp_gt.shape}")
    # make diff
    height, width = disp_gt.shape
    disp = cv2.resize(disp, (width, height), interpolation=cv2.INTER_LINEAR)
    print(f"resize disp: {disp.shape}, disp_gt: {disp_gt.shape}")
    mask = disp_gt != 0
    don_care = disp_gt == 0
    disp_diff_viz = np.abs(disp_gt - disp)
    disp_diff_viz[don_care] = 0
    if show:
        plt.imshow(disp_diff_viz)
        plt.colorbar()
        plt.show()
    disp_diff = np.abs(disp_gt[mask] - disp[mask])
    print(f"disp_diff: {disp_diff.min()} <-----> {disp_diff.max()}")
    bad_pixels = np.logical_and(
        disp_diff >= 3, abs(disp_diff / disp_gt[mask]) >= 0.05)
    print(bad_pixels)
    print(mask.sum())
    disp_error_percent = 100.0 * bad_pixels.sum() / mask.sum()
    print(disp_error_percent)
    return disp_error_percent, disp_diff_viz


def count_KITTI_flow_err(flow: np.array, fl_gt_path: str, show=False) -> tuple[float, np.array, np.array]:
    # read gt
    flow_gt = cv2.imread(str(fl_gt_path), cv2.IMREAD_UNCHANGED)
    flow_gt = cv2.cvtColor(flow_gt, cv2.COLOR_BGR2RGB)
    care = flow_gt[:, :, 2] != 0
    don_care = flow_gt[:, :, 2] == 0
    print(
        f"u_gt before divide: {flow_gt[:,:,0].min()} <-----> {flow_gt[:,:,0].max()}")
    flow_gt = (flow_gt.astype(float) - 2 ** 15) / 64.0
    print(f"original flow: {flow.shape}, flow_gt: {flow_gt.shape}")
    height, width, _ = flow_gt.shape
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    print(f"resize flow: {flow.shape}, flow_gt: {flow_gt.shape}")
    flow_diff_viz = np.abs(flow_gt[:, :, :2] - flow)
    u_diff_viz = flow_diff_viz[:, :, 0]
    u_diff_viz[don_care] = 0
    v_diff_viz = flow_diff_viz[:, :, 1]
    v_diff_viz[don_care] = 0
    print(f"u: {flow[:,:,0].min()} <-----> {flow[:,:,0].max()}")
    print(f"u_gt: {flow_gt[:,:,0].min()} <-----> {flow_gt[:,:,0].max()}")
    print(f"u_diff_viz: {u_diff_viz.min()} <-----> {u_diff_viz.max()}")
    print(f"v: {flow[:,:,1].min()} <-----> {flow[:,:,1].max()}")
    print(f"v_gt: {flow_gt[:,:,1].min()} <-----> {flow_gt[:,:,1].max()}")
    print(f"v_diff_viz: {v_diff_viz.min()} <-----> {v_diff_viz.max()}")
    if show:
        plt.imshow(u_diff_viz)
        plt.title("u diff")
        plt.colorbar()
        plt.show()
        plt.imshow(v_diff_viz)
        plt.title("v diff")
        plt.colorbar()
        plt.show()
    u_diff = u_diff_viz[care]
    v_diff = v_diff_viz[care]
    u_bad_pixels = np.logical_and(
        u_diff >= 3, abs(u_diff / flow_gt[:, :, 0][care]) >= 0.05)
    v_bad_pixels = np.logical_and(
        v_diff >= 3, abs(v_diff / flow_gt[:, :, 1][care]) >= 0.05)
    bad_pixels = np.logical_or(u_bad_pixels, v_bad_pixels)
    flow_error_percent = 100.0 * bad_pixels.sum() / care.sum()
    print(flow_error_percent)
    return flow_error_percent, u_diff_viz, v_diff_viz


def count_KITTI_scene_err(d1: np.array, d2: np.array, flow: np.array, d1_gt_path: str, d2_gt_path: str, fl_gt_path: str):
    # 這部份還沒完成，如果有需要再回來寫了
    # 2024/08/09 我回來寫了
    # 取得d1 loss
    d1e, d1_loss = count_KITTI_dissp_err(d1, d1_gt_path)
    # 取得d2 loss
    d1e, d2_loss = count_KITTI_dissp_err(d2, d2_gt_path)
    # 取得flow loss
    fle, u_loss, v_loss = count_KITTI_flow_err(flow, fl_gt_path)
    # 合併
    scene_loss_map = d1_loss + d2_loss + u_loss + v_loss
    plt.imshow(scene_loss_map)
    plt.title("scene_loss_map")
    plt.colorbar()
    plt.show()
    pass


def count_KITTI_train_loss(t0_l_path: str, t0_r_path: str, t1_l_path: str, t1_r_path: str, d1_gt_path: str, d2_gt_path: str, fl_gt_path: str, depth_gt_path: str, Q: np.array):
    # 會用train是因為沒有辦法拿到tast的groundtruth
    t0_l = cv2.cvtColor(cv2.imread(t0_l_path), cv2.COLOR_BGR2RGB)
    t0_r = cv2.cvtColor(cv2.imread(t0_r_path), cv2.COLOR_BGR2RGB)
    t1_l = cv2.cvtColor(cv2.imread(t1_l_path), cv2.COLOR_BGR2RGB)
    t1_r = cv2.cvtColor(cv2.imread(t1_r_path), cv2.COLOR_BGR2RGB)

    d1 = get_disparity(t0_l, t0_r)
    d2 = get_disparity(t1_l, t1_r)
    flow = get_flow(t0_l, t1_l)
    # _3d_t0 = reproject_depth_from_disp(d1, Q)
    # _3d_t1 = reproject_depth_from_disp(d2, Q)
    # scene = get_scene(_3d_t0, _3d_t1, flow)

    ABS_THRESH = 3.0
    # 第一對視差誤差百分比
    # d1_e = count_KITTI_dissp_err(d1, d1_gt_path)
    # 第二對影像視差誤差百分比
    # d2_e = count_KITTI_dissp_err(d2, d2_gt_path)
    # 光流誤差百分比
    fl_e = count_KITTI_flow_err(flow, fl_gt_path)
    # 場景流誤差百分比
    sf_e = count_KITTI_scene_err(
        d1, d2, flow, d1_gt_path, d2_gt_path, fl_gt_path)
    # 深度對數誤差，見KITTI官網
    d_siloge
    return d1_e, d2_e, fl_e, sf_e, d_siloge


def get_kitti_depth_path_from_id(sf_id):
    train_mapping_txt = r"E:\datasets\KITTI_sceneflow\development_kit\devkit\mapping\train_mapping.txt"
    kitti_depth_train_dir = Path(
        r"E:\datasets\KITTI_Depth Prediction\data_depth_annotated\train")
    with open(train_mapping_txt) as f:
        lines = f.readlines()
    mapping = lines[int(sf_id)]
    if lines == None:
        return None
    date, drive, num = mapping.split()
    # print(drive, num)
    depth_path = kitti_depth_train_dir / drive / "proj_depth" / \
        "groundtruth" / "image_02" / (num + ".png")
    return depth_path


def test_KITTI_loss():
    sf_id = "000010"
    KITTI_sf_training_path = Path(
        r"E:\datasets\KITTI_sceneflow\scene_flow_2015_data_set\training")
    t0_l_path = KITTI_sf_training_path / "image_2" / (sf_id + "_10.png")
    t0_r_path = KITTI_sf_training_path / "image_3" / (sf_id + "_10.png")
    t1_l_path = KITTI_sf_training_path / "image_2" / (sf_id + "_11.png")
    t1_r_path = KITTI_sf_training_path / "image_3" / (sf_id + "_11.png")
    d1_gt_path = KITTI_sf_training_path / "disp_occ_0" / (sf_id + "_10.png")
    d2_gt_path = KITTI_sf_training_path / "disp_occ_1" / (sf_id + "_10.png")
    fl_gt_path = KITTI_sf_training_path / "flow_occ" / (sf_id + "_10.png")
    depth_gt_path = get_kitti_depth_path_from_id(sf_id)
    print("t0_l_path: ", t0_l_path)
    print("t0_r_path: ", t0_r_path)
    print("t1_l_path: ", t1_l_path)
    print("t1_r_path: ", t1_r_path)
    print("d1_gt_path: ", d1_gt_path)
    print("d2_gt_path: ", d2_gt_path)
    print("fl_gt_path: ", fl_gt_path)
    print("depth_gt_path: ", depth_gt_path)
    Q = get_KITTI_Q_matrix(sf_id)
    count_KITTI_train_loss(str(t0_l_path),
                           str(t0_r_path),
                           str(t1_l_path),
                           str(t1_r_path),
                           str(d1_gt_path),
                           str(d2_gt_path),
                           str(fl_gt_path),
                           str(depth_gt_path), Q)


if __name__ == '__main__':
    # test_A13()
    # test_KITTI()
    # draw_kitti_disp()
    # draw_kitti_depth()
    # draw_kitti_flow()
    # draw_kitti_scene()
    test_KITTI_loss()
    pass

# stereo: 000010
# KITTI: 2011_09_26 2011_09_26_drive_0009_sync 0000000384
