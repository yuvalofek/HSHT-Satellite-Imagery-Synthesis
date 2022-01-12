import os
import numpy as np
import argparse
import rasterio
from concurrent.futures import ThreadPoolExecutor
import json


def parse_args():
  parser = argparse.ArgumentParser('create image pairs')
  parser.add_argument('--inputA_dir', help='input directory for image A', type=str, default='../dataset/inputA')
  parser.add_argument('--inputB_dir', help='input directory for image B', type=str, default='../dataset/inputB')
  parser.add_argument('--inputB_keyword', help='input keyword selector for dir B', type=str, default='pred')
  parser.add_argument('--inputB_dropN', help='number of inputs to drop from B to match lengths', type=int, default=1)
  parser.add_argument('--out_file', help='output file', type=str, default='../flavr_imgs.json')
  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

  # Get full paths to images in the directories
  img0 = sorted([os.path.join(args.inputA_dir, path) for path in os.listdir(args.inputA_dir) if 'img0' in path])
  img1 = sorted([os.path.join(args.inputA_dir, path) for path in os.listdir(args.inputA_dir) if 'img1' in path])
  true = sorted([os.path.join(args.inputA_dir, path) for path in os.listdir(args.inputA_dir) if 'true' in path])
  trB = sorted([os.path.join(args.inputB_dir, path) for path in os.listdir(args.inputB_dir) if args.inputB_keyword in path])
  inj = trB[args.inputB_dropN:]
  input = zip(img0, true, img1, inj)

  #print(list(input))
  #print(args.out_file)
  with open(args.out_file, 'w') as f:
    json.dump(list(input), f)
  """
  if args.split_val:
    tr, val = get_train_test(input)

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
   """

if __name__ == "__main__":
  main()
