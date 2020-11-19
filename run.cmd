::python train.py --name "ade20k" --dataset_mode "ade20k" --dataroot "F:\\Download\\ADEChallengeData2016" --checkpoints_dir "D:\\weights\\xray\\spade_ade20k"

python train.py --name "xrayspade" --dataset_mode "xray" --dataroot "D:\\tisep202006\\coronal\\train\\data" --gtroot "D:\\tisep202006\\coronal\\train\\data" --checkpoints_dir "D:\\weights\\xray\\spade_xray"