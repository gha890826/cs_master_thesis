# run in FlowFormer env
# copy and run in E:/ck/master/my_scene_flow
# python png_to_disp.py

import dis
import numpy as np
from matplotlib import pyplot as plt
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
from sympy import Q, Array
import flow_compute
import random
from ff_core.utils import flow_viz
from tqdm.contrib import tzip

plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus'] = False

MAX_SCENE_FLOW = 10

TAG_CHAR = np.array([202021.25], np.float32)


def prepare_image_and_count_disp(img_l_path, img_r_path, silence=False):
    flow = flow_compute.prepare_path_and_compute_flow(
        str(img_l_path), str(img_r_path), silence=True)
    # 將flow結果drop掉flow的y向量
    disparity = np.squeeze(np.delete(np.array(flow), 1, axis=2))
    disparity = np.abs(disparity)
    if silence:
        print(f"Range: {np.min(disparity)} <-> {np.max(disparity)}")
        plt.imshow(disparity, "gray")
        plt.show()
    # 回傳shape=(x,y), 代表視差的矩陣
    return disparity


def write_my_disp(filename, disp):
    np.save(filename, disp)
    pass


paths = list(
    Path(r"e:\datasets\penghu_vitality\png_960_align_rect_rghs").iterdir())
for i in range(len(paths)):
    path = paths[i]
    print(f"folder {i+1} of {len(paths)}: {path.name}")
    png_dir_l = path / "l"
    png_dir_r = path / "r"
    png_ls = list(png_dir_l.glob("*png"))
    png_rs = list(png_dir_r.glob("*png"))
    target_path = Path(r"e:\datasets\penghu_vitality\disp_ff_rghs") / path.name
    target_path.mkdir(parents=True, exist_ok=True)
    assert (len(png_ls) == len(png_rs))

    for png_l, png_r in tzip(png_ls[:2000], png_rs[:2000]):
        assert (png_l.name[-12:] == png_r.name[-12:])
        disp = prepare_image_and_count_disp(str(png_l), str(png_r))
        target_name = str(target_path / png_l.stem)
        write_my_disp(target_name, disp)
