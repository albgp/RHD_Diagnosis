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

def parse_preprocess(preprocess):
    preprocess_fns={
        "CAMUS": lambda x:x,
        "RSofia": lambda x:x
    }
    return preprocess_fns.get(preprocess, None)

class Dataset:
    def __init__(self, data_dir, test_dir, val_frac, preprocess):
        logging.info(f'Creating Dataset Object for {data_dir}')
        self.data_dir=data_dir 
        self.test_dir=test_dir 
        self.val_frac=val_frac 
        if preprocess:
            self.preprocess=parse_preprocess(preprocess)
        else:
            self.preprocess=lambda x: x
        self.containsTest=(test_dir!="")

    def load(self):
        logging.info(f'Loading data in {self.data_dir}')
        if not os.path.exists(self.data_dir):
            logging.error("Data directory does not exists!")

        img_dir=self.data_dir+"/imgs"
        if not os.path.exists(img_dir):
            logging.error(f"\"imgs\" dir does not exist inside data directory {self.data_dir}")
            sys.exit(1)

        mask_dir=self.data_dir+"/masks"
        if not os.path.exists(mask_dir):
            logging.error(f"\"masks\" dir does not exist inside data directory {self.data_dir}")
            sys.exit(1)

        imgs={}; masks={}

        for entry in os.scandir(img_dir):
            if entry.is_file():
                with Image.open(entry.path) as tmpimg:
                    logging.debug(f"Appending img {entry.name} to imgs")
                    imgs[entry.name]=self.preprocess(np.array(tmpimg))

        for entry in os.scandir(mask_dir):
            if entry.is_file():
                with Image.open(entry.path) as tmpimg:
                    logging.debug(f"Appending img {entry.name} to masks")
                    masks[entry.name]=self.preprocess(np.array(tmpimg))
        
        dataset={"imgs":[], "masks":[]}

        for k, v in imgs.items():
            if k not in masks:
                logging.warning(f"Image {k} not found in masks folder. Skipping this file.")
            else:
                dataset["imgs"].append(v)
                dataset["masks"].append(masks[k])
        
        self.size=len(imgs)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            dataset["imgs"], dataset["masks"], test_size=self.val_frac, random_state=42)

ds=Dataset(
           args["_data_dir"],
           args["_test_dir"],
           args["_val_frac"],
           args["_preprocess"]
        )
ds.load()
logging.info("Finished loading dataset.")

if (args["_train_unet"]):
    logging.info("Training U-Net")
    logging.info("Finished training U-Net")

if (args["_train_pspnet"]):
    logging.info("Training PSP-Net")
    PSPNet()
    logging.info("Finished taining PSP-Net")
