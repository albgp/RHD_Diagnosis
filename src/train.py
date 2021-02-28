import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import logging
from PIL import Image
from sklearn.model_selection import train_test_split
from CNN.PSPNet import PSPNet
from data_handler import data_loader

#tf.config.list_physical_devices("GPU")

curr_dir=path = Path(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Main script to train multiple autoencoder networks for a given dataset')
parser.add_argument("-d","--data_dir", dest="_data_dir", type=str, nargs=1, required=False, default=str(curr_dir.parent)+"/data/CAMUS_4CH",
    help="The folder in which data is found. It needs to contain imgs/ and masks/ subdirectories.")
parser.add_argument("--test_dir", dest="_test_dir", type=str, nargs=1, required=False, default="",
    help="The folder in which test data is found.")
parser.add_argument("-od","--output_dir", dest="_output_dir", type=str, nargs=1, required=False, default=str(curr_dir.parent)+"/models", 
    help="Output directory to save trained models")
parser.add_argument("--no-test", dest="_no_test", action="store_true", default=False, 
    help="Ommit test directory. Only train is done.")
parser.add_argument("--unet", dest="_train_unet", action="store_true", default=False,
    help="Train the U-Net autoencoder.")
parser.add_argument("--pspnet", dest="_train_pspnet", action="store_true", default=False,
    help="Train the PSPNet autoencoder.")
parser.add_argument("--val_", dest="_val_frac",type=float, default=0.2, required=False,
    help="Fraction of the training dataset used for validation.")
parser.add_argument("--preprocess", dest="_preprocess", default=None, choices=["CAMUS, RSofia"], help="Preprocessing to use when loading dataset")
args = vars(parser.parse_args())
print(args)

defaults_ds={
          "input_height":None,
          "input_width":None,
          "n_classes":4,
          "verify_dataset":True,
          "batch_size":2,
          "validate":False,
          "val_images":None,
          "val_annotations":None,
          "val_batch_size":2,
          "auto_resume_checkpoint":False,
          "load_weights":None,
          "steps_per_epoch":512,
          "val_steps_per_epoch":12,
          "gen_use_multiprocessing":False,
          "ignore_zero_class":False,
          "optimizer_name":'adam',
          "do_augment":False,
          "augmentation_name":"aug_all",
          "callbacks":None,
          "custom_augmentation":None,
          "other_inputs_paths":None,
          "preprocessing":None,
          "read_image_type":1  # cv2.IMREAD_COLOR = 1 (rgb),
                             # cv2.IMREAD_GRAYSCALE = 0,
                             # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
}

def parse_preprocess(preprocess):
    preprocess_fns={
        "CAMUS": lambda x:x,
        "RSofia": lambda x:x
    }
    return preprocess_fns.get(preprocess, None)

class Dataset:
    def __init__(self, data_dir, test_dir, val_frac, preprocess, defaults_ds):
        logging.info(f'Creating Dataset Object for {data_dir}')
        self.data_dir=data_dir 
        self.test_dir=test_dir 
        self.val_frac=val_frac 
        if preprocess:
            self.preprocess=parse_preprocess(preprocess)
        else:
            self.preprocess=lambda x: x
        self.containsTest=(test_dir!="")
        self.defaults=defaults_ds
    
    def load(self, input_height, input_width, output_height, output_width):
        img_dir=self.data_dir+"/imgs"
        mask_dir=self.data_dir+"/masks"
        self._named_data=data_loader.get_pairs_from_paths(img_dir, mask_dir)
        self._loaded_data=data_loader.image_segmentation_generator(
            img_dir, 
            mask_dir,  
            self.defaults["batch_size"],  
            self.defaults["n_classes"],
            input_height, 
            input_width,
            output_height, 
            output_width,
            read_image_type=self.defaults["read_image_type"]
        )
    
    def verify(self):
        pass


if (args["_train_unet"]):
    logging.info("Training U-Net")
    ds=Dataset(
           args["_data_dir"],
           args["_test_dir"],
           args["_val_frac"],
           args["_preprocess"],
           defaults_ds
        )
    ds.load2()
    logging.info("Finished loading dataset.")
    logging.info("Finished training U-Net")

if (args["_train_pspnet"]):
    logging.info("Training PSP-Net")
    ds=Dataset(
           args["_data_dir"],
           args["_test_dir"],
           args["_val_frac"],
           args["_preprocess"],
           defaults_ds
        )
    pspnet=PSPNet(
        {   
            "data": ds,
            "callbacks": [],
            "validate": True
            
        }
    )
    pspnet.create_model()
    ds.load2(
        pspnet.model.input_height,
        pspnet.model.input_width,
        pspnet.model.output_height,
        pspnet.model.output_width
    )
    pspnet.train()
    logging.info("Finished taining PSP-Net")
