"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import myutils

import optuna

def test(model, test_ds):
    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.eval()
    #initialize metrics
    losses, psnrs, ssims = myutils.init_meters('1*L1')
    for i, data in enumerate(test_ds):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        myutils.eval_metrics(model.real_B, model.fake_B, psnrs, ssims)
    model.train()
    return [val.avg for val in losses.values()][0], psnrs.avg, ssims.avg

def objective(trial):
    opt = TrainOptions().parse()   # get training options

    ### Define Optuna parameters
    opt.n_layers_D = trial.suggest_int('n_layers_D', 3, 5)
    opt.lr = trial.suggest_float('lr', 1e-6, 1e-3)
    opt.gan_mode = trial.suggest_categorical('gan_mode', ['vanilla', 'lsgan'])
    opt.lr_policy = trial.suggest_categorical('lr_policy', ['linear', 'step', 'cosine', 'plateau'])
    opt.lr_decay_iters = trial.suggest_int('lr_decay_iters', 40, 70)
    opt.beta1 = trial.suggest_float('beta1', 1-1e-3, 1-1e-5)
    opt.batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8])
    print('trial paramaters: ', trial.params)

    # create datasets
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training images = %d' % dataset_size)

    opt.isTraining = False
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'val'
    test_ds = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        model.epoch_loss_G = 0
        model.epoch_loss_D = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            l1, psnr, ssim = test(model, test_ds)
            #print(f'l1: {l1}, psnr {psnr}, ssim {ssim}')

        loss_G = model.epoch_loss_G/len(dataset)
        loss_D = model.epoch_loss_D/len(dataset)
        #print(f'Loss G: {loss_G}')
        #print(f'Loss D: {loss_D}')
        #print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    return psnr, ssim

def main():
  study = optuna.create_study(study_name='Pix2pixSecondary', directions=['maximize', 'minimize'])
  study.optimize(objective, n_trials=30)
  print('Best Parameters: ', study.best_trial.value)

if __name__ == '__main__':
  main()
