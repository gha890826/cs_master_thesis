# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from datasets.coco_reader.coco import build as build_coco
from datasets.voc_reader.voc import build as build_voc
from datasets.sun_reader.sun import build as build_sun
from datasets.abnormal_fish_reader.abnormal_fish import build as build_abnormal_fish
from datasets.vitality_reader.vitality import build as build_vitality

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):

    if args.dataset_file == 'coco':
        return build_coco(image_set, args)

    if args.dataset_file == 'voc':
        return build_voc(image_set, args)

    elif args.dataset_file == 'sun':
        dataset = build_sun(image_set, args)
        return dataset
    elif args.dataset_file == 'abnormal_fish':
        dataset = build_abnormal_fish(image_set, args)
        return dataset
    elif args.dataset_file == 'vitality':
        return build_vitality(image_set, args)
    
    raise ValueError(f'dataset {args.dataset_file} not supported')
