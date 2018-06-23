#!/usr/bin/env python
from __future__ import print_function
import sys
import re
import imageio
import datetime
import argparse
import glob
import gzip
import shutil
import os
import os.path
import io
import numpy as np

GAP = 20000
POINTS_IN_FILE = 20000000
INTERVAL = 10

def log(message, fle=sys.stderr):
    print(str(datetime.datetime.now()) + "==" + message, file=fle)

'''
Extracts grayscale data from bmp, and writes it as a signal
'''
def process_bmp_file(file_name, img_name, start_time, fle, scale, V, H):
    t = start_time
    img = imageio.imread(file_name)
    line_counter = 0
    log("Started " + file_name)
    stepV = 1
    stepH = 1
    if scale == True:
        stepV = 2
        stepH = 2
    if V == True:
        stepV = 2
        stepH = 1
    if H == True:
        stepV = 1
        stepH = 2

    for i in range(0, img.shape[0], stepV): #traverses through height of the image
        row = img[i]
        for j in range(0, img.shape[1], stepH): #traverses through width of the image
            #value = img[i][j][0] #since the image is grayscale, we need only one channel and the value '0' indicates just that
            value = row[j][0] #since the image is grayscale, we need only one channel and the value '0' indicates just that
            #print("%s,%s,%s,%s,%s" % (t, img_name, i, j, value))
            print("%s,%s,%s" % (t, img_name, value), file=fle)
            t += INTERVAL #arbitrary time interval between pixel values
            line_counter += 1
    log("Ended " + file_name)
    t += GAP #arbitrary time gap between images
    return (t, line_counter)

'''
Extracts grayscale data from bmp, compresses, and writes the compressed stream as a signal
'''
def process_compressed_bmp_grayscale(file_name, img_name, start_time, fle):
    t = start_time
    log("Started " + file_name)
    line_counter = 0
    img = imageio.imread(file_name)
    b_counter = 0
    byte_l = []
    for i in range(0, img.shape[0]): #traverses through height of the image
        row = img[i]
        for j in range(0, img.shape[1]): #traverses through width of the image
            byte_l.append(row[j][0])
            b_counter += 1

    container = io.BytesIO()

    with gzip.GzipFile(fileobj=container, mode="wb") as f:
        f.write(bytearray(byte_l))

    compressed = container.getvalue()

    #print("Compressed size is " + str(len(compressed)))
    #print(compressed)


    # Skip the first 30 header characters in the file
    #
    for nI in range(0,len(compressed)):
        #value = int(compressed[nI],16)
        print("%s,%s,%s" % (t, img_name, compressed[nI]), file=fle)
        t += INTERVAL #arbitrary time interval between pixel values
        line_counter += 1
    log("Ended " + file_name)
    t += GAP #arbitrary time gap between images
    return (t, line_counter)

'''
Compresses bmp file, and writes the compressed stream as a signal
'''
def process_compressed_bmp_file(file_name, img_name, start_time, fle):
    t = start_time
    TEMP_FILE = 'temp_img_a_z_0_9_f.gz'
    if (os.path.isfile(TEMP_FILE) == True):
        os.remove(TEMP_FILE)
    log("Started " + file_name)
    with open(file_name, 'rb') as f_in, gzip.open(TEMP_FILE, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    compressed = []
    line_counter = 0
    with open(TEMP_FILE, 'rb') as f_in:
        compressed = f_in.read()

    print("Compressed size is " + str(len(compressed)))

    # Skip the first 30 header characters in the file
    #
    for nI in range(30,len(compressed)):
        #value = int(compressed[nI],16)
        print("%s,%s,%s" % (t, img_name, compressed[nI]), file=fle)
        t += INTERVAL #arbitrary time interval between pixel values
        line_counter += 1
    log("Ended " + file_name)
    t += GAP #arbitrary time gap between images
    if (os.path.isfile(TEMP_FILE) == True):
        os.remove(TEMP_FILE)
    return (t, line_counter)

'''
Scales down the grayscale image.
'''
def scale_bmp_grayscale(file_name, img_name, start_time, fle, scale, method=None, gs=False):
    t = start_time
    log("Started " + file_name)
    line_counter = 0
    img = np.asarray(imageio.imread(file_name))
    b_counter = 0
    scaled = np.ndarray((3,0), dtype=np.uint8)

    for i in range(0, img.shape[0], scale): #traverses through height of the image
        for j in range(0, img.shape[1], scale): #traverses through width of the image
            cands = img[i:i+scale,j:j+scale,:3]
            if(method == None):
                scaled = np.append(scaled, cands[np.random.randint(0,cands.shape[0]),np.random.randint(0,cands.shape[1]),:].reshape(-1,1), axis = 1)
            if(method == "max"):
                scaled = np.append(scaled, np.max(cands, axis = (0,1)).reshape(-1,1), axis = 1)
            if(method == "avg"):
                scaled = np.append(scaled, np.mean(cands, axis = (0,1)).reshape(-1,1), axis = 1)

    if(gs):
        scaled = scaled[0,:][np.newaxis,:]

    for nI in range(0,scaled.shape[1]):
        print("%s,%s" % (t, img_name), end='', file=fle)
        for p in scaled[:,nI]:
            print(",%s" % (p), end='', file=fle)
        print(file=fle)
        t += INTERVAL #arbitrary time interval between pixel values
        line_counter += 1

    log("Ended " + file_name)
    t += GAP #arbitrary time gap between images
    return (t, line_counter)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--start_time", dest='start_time', type=int, default=1520000000000,
                help="Start the file at this this timestamp")
    parser.add_argument("-o", "--file_prefix", dest='file_prefix',
                help="Output file prefix name")
    parser.add_argument("-c", "--compress", dest='compress', action='store_true',
                help="Compress using gzip and extract the contents")
    parser.add_argument("-s", "--scale", dest='scale', action='store_true',
                help="Scale the image down by skipping every other pixel")
    parser.add_argument("-V", "--scale_vertical", dest='scale_v', action='store_true',
                help="Scale the image down by skipping every other row")
    parser.add_argument("-H", "--scale_horizontal", dest='scale_h', action='store_true',
                help="Scale the image down by skipping every other column")
    parser.add_argument("-f", "--files", dest='files', nargs="+",
                help="List of bmp files")
    parser.add_argument("-F", "--factor", dest='factor', type=int, default = 2,
                help="Downscaling factor (for use with -d)")
    parser.add_argument("-d", "--downscale", dest='dscale', action='store_true',
                help="Downscale the image")
    parser.add_argument("-g", "--grayscale", dest='gscale', action='store_true',
                help="Only use one color channel of the image")
    parser.add_argument("-m", "--method", dest='method', type=str, default = None,
                help="Downscaling method ('max' or 'avg'; if none is given, will randomly sample)")
    parser.set_defaults(scale=False)
    parser.set_defaults(dscale=False)
    parser.set_defaults(gscale=False)
    parser.set_defaults(scale_h=False)
    parser.set_defaults(scale_v=False)
    parser.set_defaults(compress=False)

    return parser

def main(*args):
    parser = setup_parser()
    presult = parser.parse_args()

    if (presult.file_prefix == None):
        print("file_prefix is required")
        exit

    if presult.files == None or len(presult.files) < 1:
        print("No files to process!!")
        parser.print_help()
        return

    file_prefix = presult.file_prefix
    index = 1
    start_time = presult.start_time

    log("Starting time for this output file is : " + str(start_time))
    start_time0 = start_time
    line_counter = 0
    file_counter = 2
    fle = open(file_prefix + "01.csv", "w", 16*1024)
    print("time,image" + (",pixel value" if presult.gscale else ",r value,g value,b value"), file=fle)
    tfiles = presult.files
    files = []

    #
    # Expanding wildcard characters on Windows OS
    #
    for wildcard in tfiles:
        files.extend(glob.glob(wildcard))

    for nI in range(len(files)):
        file_name = files[nI]
        #
        # Replace the path & use the file name as the batch_id
        #
        img_name = re.sub("^.*\\\\","",file_name)
        img_name = re.sub("^.*/","",img_name)
        img_name = re.sub("\.[a-zA-Z0-9]*$","",img_name)
        log("Processing file === " + file_name + ",  img_name === " + img_name)
        if (presult.compress == True):
            (start_time, lc) = process_compressed_bmp_grayscale(file_name, img_name, start_time, fle)
        elif (presult.dscale == True):
            (start_time, lc) = scale_bmp_grayscale(file_name, img_name, start_time, fle, presult.factor, presult.method, presult.gscale)
        else:
            (start_time, lc) = process_bmp_file(file_name, img_name, start_time, fle, presult.scale, presult.scale_v, presult.scale_h)
        line_counter += lc

        if (line_counter > POINTS_IN_FILE):
            fle.close()
            fle = open(file_prefix + "{0:02d}".format(file_counter) + ".csv", "w", 16*1024)
            print("time,image" + (",pixel value" if presult.gscale else ",r value,g value,b value"), file=fle)
            file_counter += 1
            line_counter = 0

    log("               Start time : " + str(start_time0))
    log("                 End time : " + str(start_time))
    log("Suggested NEXT FILE start : " + str(start_time + 2*GAP))

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        print ("Must be using Python 3")
        sys.exit()
    main(*sys.argv)
