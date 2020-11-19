"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import util.util as util
import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset_xray
import numpy as np
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms
from math import ceil, floor
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pdb


class XrayDataset(Pix2pixDataset):

    def initialize(self, opt):
        self.opt = opt

        input_paths, gt_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(input_paths)
        util.natural_sort(gt_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        try:
            table_df = pd.read_csv(str(Path(opt.dataroot).parents[1]) + '/spade_train_test_split_' + opt.dataset_mode + '.csv')
            print('Using existing train/test split')
        except:
            print('Generating new train/test split')
            val_index = [1] * (len(input_paths) - opt.nval) + [0] * opt.nval
            random.shuffle(val_index)
            table_df = pd.DataFrame(list(zip(input_paths, gt_paths, val_index)),
                                    columns=['input_paths', 'gt_paths', 'is_train'])
            for index, row in table_df.iterrows():
                assert self.paths_match(row.input_paths, row.gt_paths[:-4])

            table_df.to_csv(str(Path(opt.dataroot).parents[1]) + '/spade_train_test_split_' + opt.dataset_mode + '.csv')

        if opt.isTrain or opt.isVal:
            is_train = 1 if opt.isTrain else 0
            self.input_paths = table_df.input_paths[table_df.is_train == is_train].tolist()
            self.gt_paths = table_df.gt_paths[table_df.is_train == is_train].tolist()
        else:
            self.input_paths = input_paths

        self.instance_paths = instance_paths
        size = len(self.input_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        input_dir = opt.dataroot
        input_paths = make_dataset_xray(input_dir)

        gt_paths = []
        if opt.isTrain or opt.isVal:
            gt_dir = opt.gtroot
            gt_paths = make_dataset_xray(gt_dir)

        instance_paths = []  # don't use instance map

        return input_paths, gt_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]

        if len(filename1_without_ext) != len(filename2_without_ext):
            filename2_without_ext = filename2_without_ext[:-4]

        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Input Image
        input_path = self.input_paths[index]

        inp_arr, self.inp_min_val, self.inp_max_val, voxel_sizes, original_input_shape = \
            self.load_nii_to_arr(input_path, self.opt.phase, self.opt.input_min_max)

        input_ = Image.fromarray(inp_arr)
        input_ = self.resize_keep_ratio(input_, self.opt.load_size)
        img_pad = self.pad_image(input_, self.opt, self.inp_min_val)
        input_ = np.array(img_pad(input_))

        input_tensor = self.norm(input_, self.opt.norm_method, self.inp_min_val, self.inp_max_val,)

        gt_tensor, instance_tensor = 0, 0

        if self.opt.isTrain or self.opt.isVal:
            gt_path = self.gt_paths[index]
            gt_arr, self.out_min_val, self.out_max_val, _, _ = self.load_nii_to_arr(gt_path, self.opt.phase, self.opt.output_min_max)

            gt_ = Image.fromarray(gt_arr)
            gt_ = self.resize_keep_ratio(gt_, self.opt.load_size)
            img_pad = self.pad_image(gt_, self.opt)
            gt_ = np.array(img_pad(gt_))

            gt_tensor = self.norm(gt_, 'minmax', self.out_min_val, self.out_max_val)

        input_dict = {'label': input_tensor,
                      'instance': instance_tensor,
                      'image': gt_tensor,
                      'path': input_path,
                      'voxel_size': voxel_sizes,
                      'original_input_shape': original_input_shape,
                      }

        return input_dict

    def resize_keep_ratio(self, img, target_size):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # im.thumbnail(new_size, Image.ANTIALIAS)
        try:
            im = img.resize(new_size, Image.LANCZOS)
        except:
            im = img.resize(new_size, Image.NEAREST)
        return im

    def pad_image(self, img, opt, pad_value=-1024):
        old_size = img.size
        target_size = opt.load_size
        pad_value = -1024 if opt.phase == 'train' or opt.phase == 'val' else int(pad_value)

        pad_size_w = (target_size - old_size[0]) / 2
        pad_size_h = (target_size - old_size[1]) / 2

        if pad_size_w % 2 == 0:
            wl, wr = int(pad_size_w), int(pad_size_w)
        else:
            wl = ceil(pad_size_w)
            wr = floor(pad_size_w)

        if pad_size_h % 2 == 0:
            ht, hb = int(pad_size_h), int(pad_size_h)
        else:
            ht = ceil(pad_size_h)
            hb = floor(pad_size_h)

        return transforms.Compose(
            [
                transforms.Pad((wl, ht, wr, hb), fill=pad_value),
                # transforms.ToTensor(),
            ]
        )

    def norm(self, img, method, min_val=-1024, max_val=-1024):
        if method == 'standardization':
            normalized_a = (img - img.mean()) / img.std()
        elif method == 'prof_norm':
            mean, std = img.mean(), img.std()
            A_neg2std = np.where(img < mean - (2*std), mean - (2*std), img)
            percentile0, percentile99 = np.percentile(A_neg2std, 0), np.percentile(A_neg2std, 99)
            normalized_a = (img - percentile0) / (percentile99 - percentile0)
        elif method == 'minmax':
            normalized_a = (img - min_val) / (max_val - min_val)
        to_tensor = transforms.ToTensor()
        normalized_a = to_tensor(normalized_a)
        return normalized_a

    def load_nii_to_arr(self, path, phase, minmax):
        voxel_sizes, original_input_shape = 0, 0

        if phase == 'train' or phase == 'val':
            nii = nib.load(path)
            a_arr = np.transpose(np.array(nii.dataobj)[:, :, 0], axes=[1, 0])
            if minmax:
                a_min_val, a_max_val = minmax.split(',')
            else:
                a_min_val, a_max_val = -1024, -1024
            a_min_val, a_max_val = int(a_min_val), int(a_max_val)
        else:
            nii = nib.load(path)
            nii_shape = len(np.array(nii.dataobj).shape)

            if nii_shape == 2:
                a_arr = np.transpose(np.array(nii.dataobj), axes=[1, 0])
            elif nii_shape == 3:
                a_arr = np.transpose(np.array(nii.dataobj), axes=[2, 1, 0])[0, :, :]
            elif nii_shape == 4:
                a_arr = np.transpose(np.array(nii.dataobj), axes=[3, 2, 1, 0])[0, 0, :, :]

            header = nii.header
            voxel_sizes = header.get_zooms()
            original_input_shape = np.transpose(a_arr, axes=[1, 0]).shape

            a_min_val = int(a_arr.min())
            a_max_val = int(a_arr.max())

        return a_arr, a_min_val, a_max_val, voxel_sizes, original_input_shape
