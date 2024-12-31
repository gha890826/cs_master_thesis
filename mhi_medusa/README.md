# MHI-DETR

以MEDUSA為基底去改

## 指令

第一次訓練時以DETR checkpoint 為pretrain weight:

```cmd
python main.py --pretrain "checkpoints/medusa_800x1333.pth" --batch_size 4 --num_workers 2 --epochs 5 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

訓練100epochs的指令:

```cmd
python main.py --pretrain "checkpoints/medusa_800x1333.pth" --batch_size 4 --num_workers 2 --epochs 100 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.2" --output "output" --dataset_file vitality
```

接續訓練時以用resume繼續:

```cmd
python main.py --resume "logs\MEDUSA-resnet50-batch-4-epoch-5\checkpoint.pth" --batch_size 4 --num_workers 2 --epochs 10 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

用pretrain權重訓練澎湖

```cmd
python main.py --pretrain "checkpoints/medusa_800x1333.pth" --batch_size 4 --num_workers 2 --epochs 100 --lr_drop 100 --print_freq 200 --path "E:\datasets\penghu_vitality\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

用A13權重繼續訓練澎湖

```cmd
python main.py --resume "logs\MEDUSA-resnet50-batch-4-epoch-100_A13_pretrained_label_id_from_0\checkpoint.pth" --batch_size 4 --num_workers 2 --epochs 200 --lr_drop 100 --print_freq 200 --path "E:\datasets\penghu_vitality\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

如果要完全自己訓練，不要加入`--pretrain`變數

```cmd
python main.py --batch_size 4 --num_workers 2 --epochs 5 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

對資料集進行eval:

```cmd
python main.py --eval --resume "logs\MEDUSA-resnet50-batch-4-epoch-10\checkpoint.pth" --batch_size 4 --num_workers 2 --epochs 5 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

對資料集的eval進行檢視:

```cmd
python main.py --eval_viz --resume "logs\MEDUSA-resnet50-batch-4-epoch-100_pretrained\checkpoint.pth" --batch_size 4 --num_workers 2 --epochs 5 --lr_drop 100 --print_freq 200 --path "E:\datasets\ck_master_dataset\coco_type\vitality_0.1" --output "output" --dataset_file vitality
```

使用RGB及MHI進行視覺化:

<!-- 00000035 train -->
```cmd
python main.py --image "E:\datasets\ck_master_dataset\png_960\a13_2022-10-28-07-00-04\l\00000035.png" --mhi "E:\datasets\ck_master_dataset\mhi\a13_2022-10-28-07-00-04\00000006_00000035.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality
```

<!-- 00000050 val -->
```cmd
python main.py --image "E:\datasets\ck_master_dataset\png_960\a13_2022-10-28-07-00-04\l\00000050.png" --mhi "E:\datasets\ck_master_dataset\mhi\a13_2022-10-28-07-00-04\00000021_00000050.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality
```

<!-- 澎湖 50 val dataset -->
```cmd
python main.py --image "E:\datasets\penghu_vitality\coco_type\vitality_0.1\val_img\00000050.png" --mhi "E:\datasets\penghu_vitality\coco_type\vitality_0.1\val_mhi\00000036_00000050.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality
```
<!-- 澎湖 80 val dataset -->
```cmd
python main.py --image "E:\datasets\penghu_vitality\coco_type\vitality_0.1\val_img\00000080.png" --mhi "E:\datasets\penghu_vitality\coco_type\vitality_0.1\val_mhi\00000066_00000080.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality
```

<!-- 澎湖RGHS 50 val dataset -->
```cmd
python main.py --image "E:\datasets\penghu_vitality\coco_type\vitality_rghs_1.0_fold1\val_img\00000050.png" --mhi "E:\datasets\penghu_vitality\coco_type\vitality_rghs_1.0_fold1\val_mhi\00000036_00000050.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100_penghu_rghs_fold1\checkpoint.pth" --dataset_file vitality
```

<!-- 00001340 not in dataset -->
```cmd
python main.py --image "E:\datasets\ck_master_dataset\png_960\a13_2022-10-28-07-00-04\l\00001340.png" --mhi "E:\datasets\ck_master_dataset\mhi\a13_2022-10-28-07-00-04\00001311_00001340.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality
```

使用RGB及MHI進行並儲存結果:

<!-- 00000035 train -->
```cmd
python main.py --image "E:\datasets\ck_master_dataset\png_960\a13_2022-10-28-07-00-04\l\00000050.png" --mhi "E:\datasets\ck_master_dataset\mhi\a13_2022-10-28-07-00-04\00000021_00000050.npy" --resume "logs\MEDUSA-resnet50-batch-4-epoch-100\checkpoint.pth" --dataset_file vitality --save_dir "E:\datasets\ck_master_dataset\mhi-detr_predict\a13_2022-10-28-07-00-04"
```

## Note

2024-07-09 發現了訓練資料的label id錯誤，應該從0開始的變成從1開始，要重新訓練
2024-07-10 label id從0開始的結果不錯
