import os
import numpy as np
import argparse
import rasterio
from concurrent.futures import ThreadPoolExecutor

def parse_args():
  parser = argparse.ArgumentParser('create image pairs')
  parser.add_argument('--inputA_dir', help='input directory for image A', type=str, default='../dataset/inputA')
  parser.add_argument('--inputB_dir', help='input directory for image B', type=str, default='../dataset/inputB')
  parser.add_argument('--inputA_keyword', help='input keyword selector for dir A', type=str, default='pred')
  parser.add_argument('--inputB_keyword', help='input keyword selector for dir B', type=str, default='pred')
  parser.add_argument('--inputB_dropN', help='number of inputs to drop from B to match lengths', type=int, default=1)
  parser.add_argument('--out_dir', help='output directory', type=str, default='../dataset/test_AB')
  parser.add_argument('--bands', help='number of bands to save', type=int, default=1)
  parser.add_argument('--im_size', help='size of each image', type=int, default=128)
  parser.add_argument('--split_val', action='store_true', help='create a validation dir')
  parser.add_argument('--normalize', action='store_true', help='normalize images first')
  args = parser.parse_args()
  return args

def center_crop(im, im_size):
  """
  Center crop image to im_size
  """
  im_size = im_size//2
  return im[:,(im.shape[1]//2-im_size):(im.shape[1]//2+im_size),(im.shape[2]//2-im_size):(im.shape[2]//2+im_size)]

def load_image(path, bands, size):
    """
    Use rastrio to open up tif file, extract the desired number of bands, and center crop to right size
    """
    with rasterio.open(path) as src:
      im = src.read()[:bands]
    im = center_crop(im, size)
    return im

def image_write(path_A, path_B, path_AB, bands=1, im_size=128, normalize=True):
    """
    Get the images, concatenate them together, and save in the correct format
    """
    # get images
    im_A = load_image(path_A, bands, im_size)
    im_B = load_image(path_B, bands, im_size)

    if normalize:
      im_A = im_A/im_A.max()
      im_B = im_B/im_B.max()

    # concat and save
    im_AB = np.concatenate([im_A, im_B])
    save_image(im_AB, path_AB)

def save_image(array, path):
  """
  Save a numpy array as a tif file at a specified path
  """
  with rasterio.open(path, mode='w', driver='GTiff',
                       height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=array.dtype) as src:
    src.write(array)

def read_write_image(inp_path, out_path, bands, im_size, normalize):
  im = load_image(inp_path, bands, im_size)
  if normalize:
    im = im/im.max()
  save_image(im, out_path)

def get_train_test(paths, test_frac=0.2, random_state=12345, shuffle=True):
    """
    Returns the training and validation paths
    :param data_root: root directory for images
    :param test_frac: fraction to make into a validation set
    :param random_state: random state
    :param shuffle: whether to shuffle the input paths
    :return: training paths, test paths (lists)
    """
    from sklearn.model_selection import train_test_split

    test_frac = test_frac * 0.01 if test_frac >= 1 else test_frac
    test_size = int(test_frac * len(paths))
    tr_paths, test_paths = train_test_split(paths,
                                            test_size=test_size,
                                            random_state=random_state,
                                            shuffle=shuffle)
    return tr_paths, test_paths




def main():
  args = parse_args()
  for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

  # Get full paths to images in the directories
  trA = sorted([os.path.join(args.inputA_dir, path) for path in os.listdir(args.inputA_dir) if args.inputA_keyword in path])
  trB = sorted([os.path.join(args.inputB_dir, path) for path in os.listdir(args.inputB_dir) if args.inputB_keyword in path])
  trB = trB[args.inputB_dropN:]

  true = sorted([os.path.join(args.inputA_dir, path) for path in os.listdir(args.inputA_dir) if 'true' in path])

  if args.split_val:
    trA, valA = get_train_test(trA)
    trB, valB = get_train_test(trB)
    true, true_val = get_train_test(true)

  if not os.path.exists(args.out_dir+'/trainA'):
    print('Making train dir')
    os.makedirs(os.path.join(args.out_dir, 'trainA'))
    os.makedirs(os.path.join(args.out_dir, 'trainB'))

  path_pairs = [os.path.join(os.path.join(args.out_dir,'trainA'), f'sample{i:03}.tif') for i in range(len(trA))]
  path_true =  [os.path.join(os.path.join(args.out_dir,'trainB'), f'sample{i:03}.tif') for i in range(len(trA))]
  with ThreadPoolExecutor() as e:
    print('making trainA')
    for i, pth in enumerate(zip(trA, trB)):
      e.submit(image_write, pth[0], pth[1], path_pairs[i], bands=args.bands, im_size=args.im_size, normalize=args.normalize)
    print('making trainB')
    for pth, pth_true in zip(true, path_true):
      e.submit(read_write_image, pth, pth_true, bands=args.bands, im_size=args.im_size, normalize=args.normalize)

  if args.split_val:
    if not os.path.exists(args.out_dir+'/valA'):
      print('Making validation dir')
      os.mkdir(os.path.join(args.out_dir, 'valA'))
      os.mkdir(os.path.join(args.out_dir, 'valB'))

    path_pairs = [os.path.join(os.path.join(args.out_dir,'valA'), f'sample{i:03}.tif') for i in range(len(valA))]
    path_true = [os.path.join(os.path.join(args.out_dir,'valB'), f'sample{i:03}.tif') for i in range(len(valA))]
    with ThreadPoolExecutor() as e:
      for i, pth in enumerate(zip(valA, valB)):
        e.submit(image_write, pth[0], pth[1], path_pairs[i], bands=args.bands, im_size=args.im_size, normalize=args.normalize)
      for pth, pth_true in zip(true_val, path_true):
        e.submit(read_write_image, pth, pth_true, bands=args.bands, im_size=args.im_size, normalize=args.normalize)

if __name__ == "__main__":
  main()
