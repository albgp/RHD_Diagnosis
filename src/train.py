import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
#tf.config.list_physical_devices("GPU")

curr_dir=path = Path(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(description='Main script to train multiple autoencoder networks for a given dataset')
parser.add_argument("-d","--data_dir", type=str, nargs=1, required=False, default=str(curr_dir.parent)+"/data/CAMUS",
    help="The folder in which data is found. train/ subdirectory must exist. If test/ exists, test data will be also loaded.")
parser.add_argument("-od","--output_dir", type=str, nargs=1, required=False, default=str(curr_dir.parent)+"/models", 
    help="Output directory to save trained models")
parser.add_argument("--no-test", dest="_no_test", action="store_true", default=False, 
    help="Ommit test directory. Only train is done.")
parser.add_argument("--unet", dest="_train_unet", action="store_true", default=False,
    help="Train the U-Net autoencoder.")
parser.add_argument("--pspnet", dest="_train_pspnet", action="store_true", default=False,
    help="Train the PSPNet autoencoder.")
parser.add_argument("--preprocess", dest="_preprocess", default=None, choices=["CAMUS, RSofia"], help="Preprocessing to use when loading dataset")
args = parser.parse_args()
print(args)

#from data_handler import read_data
