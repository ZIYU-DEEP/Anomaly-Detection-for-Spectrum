import numpy as np
import glob
import sys
import os

folder = sys.argv[1]
state = sys.argv[2]
ds_spec_path = '/net/adv_spectrum/data/downsample/downsample_10/' + state + '/' + folder + '/'
rss_path = '/net/adv_spectrum/data/rss/downsample_10/' + state + '/' + folder + '/'

if not os.path.exists(rss_path):
    os.makedirs(rss_path)
    print(rss_path + ' is created')

def folder_to_rss(file_path):
    rss = []
    for filename in sorted(glob.glob(file_path + '*.txt')):
        print('Beginning to process ' + filename)
        out_file = rss_path + filename.split('/')[-1].replace('_ap.txt', '_rss.txt')
        oid = open(out_file, 'w')
        with open(filename, 'r') as f:
            for line in f:
                x = line.split()
                power = str(np.mean([float(i) for i in x])) + '\n'
                oid.write(power)
                rss.append(power)
            print(np.shape(rss))
    print('RSS of ' + file_path + ' is already constructed.')
    return 

folder_to_rss(ds_spec_path)