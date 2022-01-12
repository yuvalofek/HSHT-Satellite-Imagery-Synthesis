import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import rasterio
from sklearn.model_selection import train_test_split

im_size = 128

def get_loc_paths(loc_dir:str, ic=None)-> list:
  loc_paths = list()
  for date in sorted(os.listdir(loc_dir)):
    date_path = os.path.join(loc_dir, date)

    if not os.path.isdir(date_path):
      continue
    date_images = list()
    for image in os.listdir(date_path):
      # if specified a single ic & if the name of the image matches the ic desired
      if ic is not None:
          if ic in image:
              date_images.append(os.path.join(date_path, image))
      else:
          # if we want all ics
          date_images.append(os.path.join(date_path,image))
    loc_paths.append(date_images)
  # remove empty parts
  loc_paths = [paths for paths in loc_paths if len(paths)!=0]
  return loc_paths

def get_train_test(data_root, set_length, test_frac=0.2, ic='modis', random_state=None, shuffle=True):
  paths = get_loc_paths(data_root, ic)
  test_frac= test_frac*0.01 if test_frac>=1 else test_frac
  test_size = int(test_frac*len(paths))
  tr_idx, test_idx = train_test_split(list(range(len(paths)-set_length)), # all idx but last set_length-1
                                      test_size=test_size,
                                      random_state=random_state,
                                      shuffle=shuffle)
  return paths, tr_idx, test_idx

class SatelliteLoader(Dataset):
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

        self.inter_frames = inter_frames
        self.n_inputs = n_inputs
        self.set_length = (n_inputs-1)*(inter_frames+1)+1 ## We require these many frames in total for interpolating `interFrames` number of
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
        img_paths = [self.paths[i+self.idx[index]] for i in range(self.set_length)]

        # Load images as tensors
        images = list()
        for pth in img_paths:
          with rasterio.open(pth[0]) as src:
            images.append(torch.from_numpy(src.read(out_dtype='float')[:self.channels]).type(torch.FloatTensor))

        # apply transformations if training
        seed = random.randint(0, 2**32)
        images_ = []
        if self.training:
            for img_ in images:
                # Apply the same transformation by using the same seed
                random.seed(seed)
                images_.append(self.transforms(img_))
            # Random Temporal Flip
            if random.random() >= 0.5:
                images_ = images_[::-1]
        else:
            # ensure sizes match with a crop
            for img_ in images:
                # Apply the same transformation by using the same seed
                random.seed(seed)
                images_.append(self.transforms(img_))
        images = images_
        # pick out every inter_frame+1 images as inputs
        inp_images = [images[idx] for idx in range(0, self.set_length, self.inter_frames+1)]
        rem = self.inter_frames%2
        gt_images = [images[idx] for idx in range(self.set_length//2-self.inter_frames//2 , self.set_length//2+self.inter_frames//2+rem)]
        return inp_images, gt_images

    def __len__(self):
        return len(self.idx)

def get_loader(paths, idx, batch_size, shuffle, num_workers, is_training=True, inter_frames=3, n_inputs=4, channels=3):
    dataset = SatelliteLoader(paths, idx , is_training, inter_frames=inter_frames, n_inputs=n_inputs, channels=channels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
