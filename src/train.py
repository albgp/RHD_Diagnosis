import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import logging
from PIL import Image
from sklearn.model_selection import train_test_split
from CNN.PSPNet import PSPNet
from data_handler import data_loader
from params import defaults_ds
#tf.config.list_physical_devices("GPU")
from cli_train import *
logging.basicConfig(level=logging.INFO)

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
    ds.load()
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
    ds.load(
        pspnet.model.input_height,
        pspnet.model.input_width,
        pspnet.model.output_height,
        pspnet.model.output_width
    )
    pspnet.train()
    logging.info("Finished taining PSP-Net")
