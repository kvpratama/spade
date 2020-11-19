"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import data
import numpy as np
import nibabel as nib
from skimage.measure import label
import pandas as pd

import matplotlib.pyplot as plt
import pdb


def get_biggest_connected_region(gen_lung, n_region=2):
    """ return n_biggest connected region -> similar to region growing in Medip """
    labels = label(gen_lung)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    biggest_regions_index = (-n_connected_region).argsort()[1:n_region + 1]  # get n biggest regions index without BG

    biggest_regions = np.array([])
    for ind in biggest_regions_index:
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            biggest_regions += labels == ind
    return biggest_regions


if __name__ == '__main__':

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.phase = 'test'

    data_loader = data.create_dataloader(opt)

    epoch_list = opt.test_epoch.split(',')

    b_min_val, b_max_val = opt.output_min_max.split(',')
    b_min_val, b_max_val = int(b_min_val), int(b_max_val)

    for epoch in epoch_list:
        lung_info = []
        opt.which_epoch = epoch
        os.makedirs("%s/%s/%s/%s/%s" % (opt.checkpoints_dir, opt.name, 'output', opt.results_dir, opt.which_epoch),
                    exist_ok=True)

        # test
        print('Testing from checkpoint %s' % opt.which_epoch)
        model = Pix2PixModel(opt)
        model.eval()

        for i, data in enumerate(data_loader):
            generated = model(data, mode='inference')

            generated_np = generated.detach().cpu().numpy()

            if opt.norm_method == 'standardization':
                denormalize_gen = generated_np * data['std'].cpu().numpy() + data['mean'].cpu().numpy()
            else:
                denormalize_gen = generated_np * (b_max_val - b_min_val) + b_min_val

            filename = data['path'][0].split("\\")[-1]  # .split(".")[0]
            filename = filename[0:filename.rfind(".")]

            # apply threshold
            denormalize_gen = np.where(denormalize_gen < opt.threshold, -1024, denormalize_gen)
            denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)

            if opt.get_lung_area:
                # find connected region
                denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask)
                connected_lung = np.where(denormalize_gen_mask, denormalize_gen[0, 0], -1024)
                denormalize_gen = connected_lung[np.newaxis, np.newaxis]

                original_width = data['original_input_shape'][0].numpy()[0]
                original_height = data['original_input_shape'][1].numpy()[0]
                ratio = float(2048) / max(original_width, original_height)

                pixel_size_resize_w = data['voxel_size'][0].numpy()[0] / ratio
                pixel_size_resize_h = data['voxel_size'][1].numpy()[0] / ratio

                area = sum(denormalize_gen_mask.flatten())

                area = area * pixel_size_resize_w * pixel_size_resize_h / 100

                lung_info.append(
                    [filename, original_width, original_height, ratio, data['voxel_size'][0].numpy()[0],
                     data['voxel_size'][1].numpy()[0],
                     pixel_size_resize_w, pixel_size_resize_h,
                     area, opt.threshold])

            nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
            # nii_np = np.transpose(connected_lung[np.newaxis, np.newaxis], axes=[3, 2, 1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            nii.header['pixdim'] = pixel_size_resize_w
            nib.save(nii, "%s/%s/%s/%s/%s/%s_fake.nii" %
                     (opt.checkpoints_dir, opt.name, 'output', opt.results_dir, opt.which_epoch, filename))

            print("%d/%d test files saved successfully" % (i + 1, len(data_loader)))

        log_df = pd.DataFrame(lung_info, columns=['filename', 'original_width', 'original_height', 'ratio_to_2048',
                                                  'original_pixel_size_w', 'original_pixel_size_resize_h',
                                                  'pixel_size_resize_w', 'pixel_size_resize_h', 'area',
                                                  'threshold'])
        log_df.to_csv("%s/%s/%s/%s/%s/area.csv" %
                      (opt.checkpoints_dir, opt.name, 'output', opt.results_dir, opt.which_epoch))

