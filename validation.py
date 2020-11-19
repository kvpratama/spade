import os
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import numpy as np
import pandas as pd
import nibabel as nib
import math
import glob
from skimage.measure import compare_ssim
import data

import matplotlib.pyplot as plt
import pdb


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def psnr(mse, pixel_max=1.0):
    if mse == 0:
        return 100
    return 20 * math.log10(pixel_max / math.sqrt(mse))


if __name__ == '__main__':

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.phase = 'val'

    data_loader = data.create_dataloader(opt)

    epoch_path = glob.glob(os.path.join(opt.checkpoints_dir, opt.name, '*G.pth'))
    epoch_filename = [os.path.basename(epoch) for epoch in epoch_path]
    epoch_list = [ep[0:ep.find('_')] for ep in epoch_filename]

    if opt.norm_method != 'prof_norm':
        a_min_val, a_max_val = opt.input_min_max.split(',')
        a_min_val, a_max_val = int(a_min_val), int(a_max_val)

    b_min_val, b_max_val = opt.output_min_max.split(',')
    b_min_val, b_max_val = int(b_min_val), int(b_max_val)

    metric = []
    for epoch in epoch_list:
        opt.which_epoch = epoch
        os.makedirs(opt.checkpoints_dir + "/" + opt.name + '/' + opt.results_dir + '/' + opt.which_epoch, exist_ok=True)

        print('Testing from checkpoint %s' % opt.which_epoch)
        model = Pix2PixModel(opt)
        model.eval()

        metric_mse, metric_psnr, metric_ssim = 0, 0, 0
        metric_epoch = []
        for i, data in enumerate(data_loader):
            generated = model(data, mode='inference')

            generated_np = generated.detach().cpu().numpy()
            xray_ = data['label'].cpu().detach().numpy()
            gt_ = data['image'].cpu().detach().numpy()

            if opt.norm_method == 'standardization':
                denormalize_gen = generated_np * data['std'].cpu().numpy() + data['mean'].cpu().numpy()
            else:
                denormalize_gen = generated_np * (b_max_val - b_min_val) + b_min_val
                # denormalize_gt = gt_ * (b_max_val - b_min_val) + b_min_val
                # if not opt.profnorm:
                #     denormalize_xray = xray_ * (a_max_val - a_min_val) + a_min_val

            mse_ = mse(data['image'].cpu().numpy()[0, 0, :, :], generated.cpu().numpy()[0, 0, :, :].astype('float64'))
            psnr_ = psnr(mse_)
            ssim = compare_ssim(data['image'].cpu().numpy()[0, 0, :, :], generated.cpu().numpy()[0, 0, :, :].astype('float64'),
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            metric_mse += mse_
            metric_psnr += psnr_
            metric_ssim += ssim

            metric_epoch.append((os.path.basename(data['path'][0]), mse_, psnr_, ssim))

            if opt.saveoutput:
                nii_np = np.transpose(denormalize_gen[0, :, :, :], axes=[2, 1, 0])
                nii = nib.Nifti1Image(nii_np, affine=None)
                nib.save(nii, opt.checkpoints_dir + "/" + opt.name + '/' + opt.results_dir + '/' + opt.which_epoch +
                         '/' + os.path.basename(data['path'][0].split('.')[0]) + '_gen.nii')

            print("%d/%d test files saved successfully" % (i + 1, len(data_loader)))

        logepoch_df = pd.DataFrame(metric_epoch, columns=['id', 'mse', 'psnr', 'ssim'])
        logepoch_df.to_csv(os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir, epoch + ".csv"))
        metric.append((epoch, metric_mse/len(data_loader), metric_psnr/len(data_loader), metric_ssim/len(data_loader)))

    log_df = pd.DataFrame(metric, columns=['epoch', 'mse', 'psnr', 'ssim'])
    log_df.to_csv(os.path.join(opt.checkpoints_dir, opt.name,  opt.results_dir, "validation.csv"))
