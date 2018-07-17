import imageio
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch, torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
import glob
import sys
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--start_time", dest='start_time', type=int, default=1520000000000,
                help="Start the file at this timestamp")
    parser.add_argument("-o", "--file_prefix", dest='file_prefix',
                help="Output file prefix name for the csv data")
    parser.add_argument("-m", "--model_prefix", dest='model_prefix',
                help="File prefix of the model files to use (FILE.mdl and FILE.stats)")
    parser.add_argument("-f", "--files", dest='files', nargs = '+',
                help="List of image files to featurize. Wildcards like *.jpg are legal")
    parser.add_argument("-e", "--leave_ext", dest='leave_ext', default = False, action= "store_true",
                help="Indicates to include the file extension in the batch name. Only use if the data is also stored that way in the facts file")
    parser.add_argument("-H", "--hidden_units", dest='h', type=int, default = 50,
                help="The number of hidden units, which determines the number of output signals when featurizing and image")
    parser.add_argument("-s", "--series", dest='series', default = False, action= "store_true",
                help="Outputs each image as a series of points instead of parallel signals")
    return parser

def write_point(signals, image, t, series, fle):
    for i in range(len(signals)):
        print(f"{t},{image},{signals[i].item()}", end = '', file = fle)
        if(not series): print(f",sig{i}", end = '', file = fle)
        else: t+=10
        print(file=fle)
    return t

def main(*args):
    parser = setup_parser()
    presult = parser.parse_args()

    if (presult.file_prefix == None):
        print("file_prefix is required")
        return
    
    if (presult.model_prefix == None):
        print("file_prefix is required")
        return

    if presult.files == None or len(presult.files) < 1:
        print("No files to process!!")
        parser.print_help()
        return

    model_prefix = presult.model_prefix
    model = torchvision.models.resnet18()
    H=presult.h
    model.fc = nn.Linear(512,H)
    model.load_state_dict(torch.load(f"{model_prefix}.mdl"))
    model = model.to(device)
    stats = torch.load(f'{model_prefix}.stats')
    mean = stats[0]
    std = stats[1]

    # fle = open("features.csv", "w")
    leave_ext = presult.leave_ext

    transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
    ]) 
    
    tfiles = presult.files
    files = []
    for wildcard in tfiles:
            files.extend(glob.glob(wildcard))
    with open(f"{presult.file_prefix}.csv", "w") as fle:
        if(presult.series): print("time,image,pixel value", file = fle)
        else: print("time,image,pixel value,signal", file = fle)
        t = presult.start_time
        count = 0
        for f in files:
            img = Image.open(f).convert('RGB')
            f = os.path.basename(f)
            if(not leave_ext): f = os.path.splitext(f)[0]
            t = write_point(model(transform(img).unsqueeze_(0).to(device))[0], f, t, presult.series, fle)
            t+= 20
            count+=1
            if(count%100 == 0): print(f"Done processing {count} images")

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        print ("Must be using Python 3")
        sys.exit()
    main(*sys.argv)