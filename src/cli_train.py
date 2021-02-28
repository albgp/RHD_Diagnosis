import argparse
from pathlib import Path
import os

curr_dir=path = Path(os.path.dirname(os.path.abspath(__file__)))

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