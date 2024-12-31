# run new train from here by:
# $ python eval_viz_vitality_fish.py
import os
import torch

batch_size = '4'
image_height = 300
image_width = 500
num_workers = '2'
epochs = '5'
lr_drop = 100
print_freq = 200
path = r"E:\datasets\ck_master_dataset\coco_type\vitality_0.1"
output = r'output'
dataset_file = 'vitality'
resume = r"logs/MEDUSA-resnet50-batch-1-epoch-10/checkpoint.pth"


conf = f' --eval_viz --resume {resume} --batch_size {batch_size} --num_workers {num_workers} --epochs {epochs} --lr_drop {lr_drop} --print_freq {print_freq} --path {path} --output_dir {output} --dataset_file {dataset_file}'

command = 'python main.py %s' % (conf)

print(command)
os.system(command)
