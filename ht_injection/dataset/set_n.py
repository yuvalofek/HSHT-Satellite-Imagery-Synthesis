import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import rasterio
from sklearn.model_selection import train_test_split
import json

im_size = 128

def get_train_test(data_root, test_frac=0.2, ic='modis', random_state=None, shuffle=True):
  with open(data_root, 'r') as path_file:
    paths = json.load(path_file)
  test_frac= test_frac*0.01 if test_frac>=1 else test_frac
  test_size = int(test_frac*len(paths))
  tr_idx, test_idx = train_test_split(list(range(len(paths))), # all idx but last set_length-1
                                      test_size=test_size,
                                      random_state=random_state,
                                      shuffle=shuffle)
  return paths, tr_idx, test_idx

class JsonLoader(Dataset):
    def __init__(self, paths, idx, is_training, inter_frames=3, n_inputs=4, channels=None):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
        """
        super().__init__()
        self.paths = paths
        self.idx = idx
        self.training = is_training
        self.channels = channels

        #self.inter_frames = inter_frames
        #self.n_inputs = n_inputs
        #self.set_length = (n_inputs-1)*(inter_frames+1)+1 ## We require these many frames in total for interpolating `interFrames` number of
                                                ## intermediate frames with `n_input` input frames.
        self.transforms = None
        if self.training:
            self.transforms =  transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.CenterCrop(im_size)
                #transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                #transforms.ToTensor()
            ])

        else:
            self.transforms = transforms.CenterCrop(im_size) #256

    def __getitem__(self, index):
        # get the paths corresponding to the images needed from the index
        img_paths = self.paths[index]

        # Load images as tensors
        images = list()
        for pth in img_paths:
          with rasterio.open(pth) as src:
            img = torch.from_numpy(src.read(out_dtype='float')[:self.channels]).type(torch.FloatTensor)
            img = img/img.max()
            images.append(img)

        # apply transformations if training
        seed = random.randint(0, 2**32)
        images_ = []

        for img_ in images:
          # Apply the same transformation by using the same seed
          random.seed(seed)
          images_.append(self.transforms(img_))

        # Remove the injected image - saved at end of list
        inj = images_.pop()

        if self.training:
            # Random Temporal Flip
            if random.random() >= 0.5:
                images_ = images_[::-1]
        images = images_


        #inp_images = [images[idx] for idx in range(0, self.set_length, self.inter_frames+1)]
        inp_images = [images[0], images[-1]]

        #rem = self.inter_frames%2
        #gt_images = [images[idx] for idx in range(self.set_length//2-self.inter_frames//2 , self.set_length//2+self.inter_frames//2+rem)]
        gt_images  = [images[1]]
        return inp_images, gt_images, inj

    def __len__(self):
        return len(self.paths)

def get_loader(paths, idx, batch_size, shuffle, num_workers, is_training=True, inter_frames=3, n_inputs=4, channels=3):
    dataset = JsonLoader(paths, idx , is_training, inter_frames=inter_frames, n_inputs=n_inputs, channels=channels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
