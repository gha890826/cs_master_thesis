# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_reader.coco_eval import CocoEvaluator
import matplotlib.pyplot as plt
import cv2
import numpy as np
from util.misc import NestedTensor
from PIL import Image


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch, max_norm=0, print_freq=50, n_iter_to_acc=1):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    batch_idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        losses.backward()
        if (batch_idx + 1) % n_iter_to_acc == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        batch_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, print_freq=50):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target,
               output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b


@torch.no_grad()
def evaluate_viz(model, criterion, postprocessors, data_loader, base_ds, device, print_freq=50):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter=", ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        print(f"rgb.shape: {samples.rgb.shape}")
        print(f"depth.shape: {samples.depth.shape}")
        print(f"depth_map.shape: {samples.depth_map.shape}")
        print(f"mask.shape: {samples.mask.shape}")

        outputs = model(samples)

        # samples有batch_size張
        print(f"samples.rgb.shape:{samples.rgb.shape}")
        # [batch, rgb(3), h, w]
        print("model outputs:\n", outputs.keys(), [
              outputs['pred_logits'].shape, outputs['pred_boxes'].shape, len(outputs['aux_outputs'])])
        loss_dict = criterion(outputs, targets)
        print("criterion output:\n", loss_dict.keys())
        weight_dict = criterion.weight_dict

        # move output to cpu
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        # 以下先拿samples[img_id]來做
        img_id = 1
        print(outputs['pred_logits'][img_id])

        # count thresh
        probas = outputs['pred_logits'].softmax(-1)[img_id, :, :-1]
        print(probas)
        # probas is torch.Tensor

        # 測試列印出所有結果(包含低信心的
        for i in outputs['pred_logits'].softmax(-1)[img_id]:
            # print(i)
            max_idx = np.argmax(i)
            max_val = i[max_idx]
            print(f"辨識結果: {max_idx}, 機率: {max_val}")

        # threshold the confidence
        keep = probas.max(-1).values > 0.5
        # print(f"keep: {keep}")

        ori_size = targets[img_id]["orig_size"]
        print(f"ori_size: {ori_size}")
        # 拿取RGB
        image = samples.rgb[img_id].cpu().numpy()
        image = cv2.merge([image[2], image[1], image[0]])
        print(f"image: {image.shape}")
        plt.imshow(image)
        plt.show()

        # 調整bbox大小
        print(f"bbox調整前: {outputs['pred_boxes'][img_id, keep]}")
        bboxes_scaled = rescale_bboxes(
            outputs['pred_boxes'][img_id, keep], (ori_size[1], ori_size[0]))
        print(f"bbox調整後: {bboxes_scaled}")

        # 保留信心度以上的
        probas = probas[keep].cpu().data.numpy()

        print(f"detect obj: {probas.shape}")
        print(probas)

        if len(bboxes_scaled) == 0:
            print("NO OBJECT TETECTED!!!")
            continue

        # 繪製框框
        for idx, (box, prob) in enumerate(zip(bboxes_scaled, probas)):
            max_idx = np.argmax(prob)
            print(max_idx)
            max_val = prob[max_idx]
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            bbox = np.array([
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ])
            bbox = bbox.reshape((4, 2))
            cv2.polylines(image, [bbox], True, (0, 255, 0), 2)
        plt.imshow(image)
        plt.show()

        # 以下MEDUSA原本
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # print(f"results len: {len(results)}")
        # print(
        #     f"what in results[0]:\n{results[0].keys()} {[i.shape for i in results[0]]}")

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target,
               output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


@torch.no_grad()
def viz(model, criterion, image_path: str, mhi: str, device, args) -> None:
    from datasets.vitality_reader.vitality import make_coco_transforms
    from pathlib import Path
    model.eval()
    criterion.eval()

    # prepare input
    mhi = np.load(mhi)
    RGB_mhi = np.delete(np.array(mhi), [3, 4, 5], axis=2)
    # print(f"before: {np.min(RGB_mhi)} <---> {np.max(RGB_mhi)}")
    RGB_mhi = RGB_mhi - np.min(RGB_mhi)
    RGB_mhi = (RGB_mhi * (255 / np.max(RGB_mhi))).round().astype(np.uint8)
    # print(f"after: {np.min(RGB_mhi)} <---> {np.max(RGB_mhi)}")
    RGB_mhi = Image.fromarray(RGB_mhi)
    w, h = RGB_mhi.size

    XYZ_mhi = np.delete(np.array(mhi), [0, 1, 2], axis=2)
    # print(f"before: {np.min(XYZ_mhi)} <---> {np.max(XYZ_mhi)}")
    XYZ_mhi = XYZ_mhi - np.min(XYZ_mhi)
    XYZ_mhi = (XYZ_mhi * (255 / np.max(XYZ_mhi))).round().astype(np.uint8)
    # print(f"after: {np.min(XYZ_mhi)} <---> {np.max(XYZ_mhi)}")
    XYZ_mhi = Image.fromarray(XYZ_mhi)
    # rgb_feature, gray_depth_feature, target
    image_size = (args.image_height, args.image_width)
    _transforms = make_coco_transforms("val", image_size)
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    img, depth, dummy_target = _transforms(RGB_mhi, XYZ_mhi, dummy_target)

    # reconstruct original depth map to be used as it is
    max = torch.max(depth[0])
    min = torch.min(depth[0])
    depth_map = (depth[0] - min) / (max - min)

    # rgb_feature, depth_feature, normalized depth_map, target
    # return img, depth, depth_map, target
    # rgb
    # xyz
    dummy_mask = torch.zeros(img.shape[-2:])

    img = img.unsqueeze(0)
    depth = depth.unsqueeze(0)
    depth_map = depth_map.unsqueeze(0)
    dummy_mask = dummy_mask.unsqueeze(0)

    print(f"img.shape: {img.shape}")
    print(f"depth.shape: {depth.shape}")
    print(f"depth_map.shape: {depth_map.shape}")
    print(f"dummy_mask.shape: {dummy_mask.shape}")

    samples = NestedTensor(img, depth, depth_map, dummy_mask)
    samples = samples.to(device)
    outputs = model(samples)
    # move output to cpu
    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
    # count thresh
    # print(outputs['pred_logits'][0])
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # print(probas)
    # threshold the confidence
    keep = probas.max(-1).values > args.thres

    # 讀取RGB照片
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    print(f"original image shape: {image.shape}")

    # 調整bbox大小 bboxes_scaled use (w,h)
    # print(f"bbox調整前: {outputs['pred_boxes'][0, keep]}")
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep], (image.shape[1], image.shape[0]))
    # print(f"bbox調整後: {bboxes_scaled}")

    # 保留信心度以上的
    probas = probas[keep].cpu().data.numpy()

    print(f"detect obj: {probas.shape}")
    # print(outputs['pred_logits'][0, keep])
    # print(probas)

    if len(bboxes_scaled) == 0:
        print("NO OBJECT TETECTED!!!")
        return

    def pick_color(n):
        if n == 0:
            # strong
            r, g, b = 255, 0, 0
        elif n == 1:
            # normal
            r, g, b = 0, 255, 0
        elif n == 2:
            # weak
            r, g, b = 0, 0, 255
        return r, g, b

    def get_label(n):
        if n == 0:
            # strong
            return "strong"
        elif n == 1:
            # normal
            return "normal"
        elif n == 2:
            # weak
            return "weak"

    # 繪製框框
    for idx, (box, prob) in enumerate(zip(bboxes_scaled, probas)):
        max_idx = np.argmax(prob)
        prob_val = prob[max_idx]
        print(max_idx, prob)
        label_clolr = pick_color(max_idx)
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ])
        bbox = bbox.reshape((4, 2))
        cv2.polylines(image, [bbox], True, label_clolr, 2)
        cv2.putText(image, f"{get_label(max_idx)} {str(round(prob_val,2))}", (bbox[3][0] + 5, bbox[3][1] - 5), cv2.FONT_HERSHEY_TRIPLEX,
                    0.5, label_clolr, 1, cv2.LINE_AA)
    if args.save_dir:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_path = Path(args.save_dir) / (Path(image_path).stem + ".png")
        cv2.imwrite(str(save_path), image)
    else:
        plt.imshow(image)
        plt.show()
