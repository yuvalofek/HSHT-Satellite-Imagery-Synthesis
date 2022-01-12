# Test dir with pred and true using a variety of metrics

from sewar import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, vifp, psnrb
import os
import rasterio
import argparse
from collections import defaultdict
import concurrent.futures as cp
import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_dir', type=str, help='directory to test. Formatting $index$pred or $index$test')
  parser.add_argument('--verbose', action='store_true', help='Print out results from individual tests')
  return parser.parse_args()


# max obtained from google earth engine
def psnr_(x,y):
  return psnr(x,y,MAX=65455)

def ssim_(x,y):
  return ssim(x,y,MAX=65455)


tests= [mse, rmse, psnr_, uqi, ssim_, ergas, scc, rase, sam, vifp, psnrb]


def evaluate(test_dir, verbose=False):
  global tests

  # get image paths
  pred_pths = [os.path.join(test_dir,im) for im in sorted(os.listdir(test_dir)) if 'pred' in im]
  true_pths = [os.path.join(test_dir,im) for im in sorted(os.listdir(test_dir)) if 'true' in im]
  assert len(pred_pths) == len(true_pths)

  # run through the images in pairs
  results= defaultdict(lambda: 0)
  for i, (pred, true) in enumerate(zip(pred_pths, true_pths)):
    # read in the images
    with rasterio.open(pred, 'r') as src:
      pred_im = src.read()
    with rasterio.open(true, 'r') as src:
      true_im = src.read()

    # for test the images using each of the tests
    for test in tests:
      try:
        result = test(true_im, pred_im)
        if verbose:
          print(f'Iteration {i} - test: {test.__name__} = {result}')
        if not np.isnan(result):
          results[test.__name__] += result
      except:
        if verbose:
          print(f'Iteration {i} - test: {test.__name__} Failed')

  # average results across all images
  results = {key: val/len(true_pths) for key, val in results.items()}
  print(results)
  return results


def main():
  args = parse_args()
  evaluate(args.test_dir, args.verbose)


if __name__ == "__main__": main()
