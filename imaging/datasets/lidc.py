
import numpy as np

def load_lidc_images(applyMask=False):
    # load the numpy file
    lidc_images = np.load('/home/matt/Research/20221207_lidc/data/lidc_nodule_database/images.npy')
    if applyMask:
        lidc_masks = np.load('/home/matt/Research/20221207_lidc/data/lidc_nodule_database/masks.npy')
        lidc_images = lidc_images * lidc_masks
    lidc_images = lidc_images.reshape(-1, 1, 64, 64)
    lidc_images = lidc_images - 1000 # Hounsfield units
    return lidc_images
