# import flowformer元件
from ast import List, Tuple
import sys

from sympy import root
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg
from core.utils.misc import process_cfg
import core.datasets
from core.utils import flow_viz
from core.utils import frame_utils
import cv2
import math
import os.path as osp
from pathlib import Path

from core.FlowFormer import build_flowformer

from core.utils.utils import InputPadder, forward_interpolate
import itertools

import flow_compute
import torch.utils.data as data

TRAIN_SIZE = [432, 960]
plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus'] = False


# ===== 載入訓練資料 =====
# 讀取深度
def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth


class KITTI_Depth_Dataset(data.Dataset):
    def __init__(self, KITTI_path=r'E:\datasets\KITTI', depth_paths=r'E:\datasets\KITTI_Depth Prediction\data_depth_annotated', type="train"):
        self.depth_02_paths = []
        self.depth_03_paths = []
        self.image_02_paths = []
        self.image_03_paths = []
        depth_paths = Path(depth_paths)
        KITTI_path = Path(KITTI_path)

        depth_paths /= type
        depth_paths = depth_paths.iterdir()
        for path in depth_paths:
            # 取得深度影像
            path_depth_02s = sorted(
                (path / "proj_depth" / 'groundtruth' / "image_02").glob("*.png"))
            path_depth_03s = sorted(
                (path / "proj_depth" / 'groundtruth' / "image_03").glob("*.png"))

            # 取得深度影像日期、車次
            drive = path.name
            date = drive[:10]
            # print(date, drive)
            # 取得RGB影像
            path_02s = sorted((KITTI_path / date / drive /
                              "image_02" / "data").glob("*.png"))[5:-5]
            path_03s = sorted(
                (KITTI_path / date / drive / "image_03" / "data").glob("*.png"))[5:-5]

            if len(path_02s) != len(path_depth_02s):
                print(
                    f"data at {drive} is not the same! {len(path_02s)} vs {len(path_depth_02s)}")
                # print(path_depth_02s)
                # print(path_02s)
                continue

            # 將資料加入陣列
            for path_depth_02, path_depth_03 in zip(path_depth_02s, path_depth_03s):
                self.depth_02_paths.append(str(path_depth_02))
                self.depth_03_paths.append(str(path_depth_03))
            for path_02, path_03 in zip(path_02s, path_03s):
                self.image_02_paths.append(str(path_02))
                self.image_03_paths.append(str(path_03))

        print(f"Add {len(self.depth_02_paths)} depth_02 path")
        print(f"Add {len(self.depth_03_paths)} depth_03 path")
        print(f"Add {len(self.image_02_paths)} image_02 path")
        print(f"Add {len(self.image_03_paths)} image_03 path")
        self.n_samples = len(self.depth_02_paths)

    def __getitem__(self, index):

        return self.image_02_paths[index], self.image_03_paths[index], self.depth_02_paths[index]

    def __len__(self):

        return self.n_samples


my_kitti = KITTI_Depth_Dataset()


# ===== 測試計算光流 =====
path1, path2, depth_path = my_kitti[0]
image1, image2 = flow_compute.prepare_image(path1, path2, keep_size=True)
depth = depth_read(depth_path)
print(f"image1: {image1.shape}, {path1}")
print(f"image2: {image2.shape}, {path2}")
print(f"depth: {depth.shape}, {depth_path}")
with torch.no_grad():
    flow = flow_compute.compute_flow(image1, image2)
print(f"flow: {flow.shape}")
