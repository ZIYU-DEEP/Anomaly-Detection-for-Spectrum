import numpy as np
import sys
import os
import glob

ori_folder = '/net/adv_spectrum/data/downsample/downsample_10/normal/' + sys.argv[1] + '/'
so_folder = '/net/adv_spectrum/data/downsample/downsample_10/abnormal/' + sys.argv[1] + '_sigOver/'
if not os.path.exists(so_folder):
    os.makedirs(so_folder)
    print(so_folder + ' is created')
start_freq = float(sys.argv[2])
bandwidth = float(sys.argv[3])
period = float(sys.argv[4])

def add_spec(filename, start_freq, bandwidth, period):
    ## Input: filename of the downsample version of Signals 
    ## Start_freq: the location where the freq is started in MHz
    ## bandwidth: channel bandwidth, in MHz
    ## period: the signal injection period, in ms
    
    time_step = 1000
    Slen = 128
    start_point = 0 # in lines, this is the startline (in time domain)
    line_count = 0
    
    fid = open(filename,'r')
    out_filename = so_folder + filename.split('/')[-1].replace('ap.txt', 'sigOver_ap.txt')
    oid = open(out_filename, 'a')
    
    # skip the former start_point lines
    for i in range(start_point):
        fid.readline()
        line_count += 1
    
    # modify the targetted slots
    flag = 0 # to indicate the end of file
    while(True):
        for i in range(int(period / 0.256)):
            ori_slot = fid.readline()
            if ori_slot != '':
                oid.write(ori_slot)
                line_count += 1
            else:
                flag = 1
                break
        
        if flag == 1:
            break
        # modify specific slot according to the start_freq and channel bandwidth
        mod_slot = [float(j) for j in fid.readline().split()]
        if np.shape(mod_slot)[0] == 128:
            line_count += 1
    #         print(int(start_freq / 5 * Slen), int((start_freq + bandwidth) / 5 * Slen))
    #         print(mod_slot[int(start_freq / 5 * Slen): int((start_freq + bandwidth) / 5 * Slen)])
            for k in range(int(start_freq / 5 * Slen), int((start_freq + bandwidth) / 5 * Slen)):
                mod_slot[k] += 3
                # try:
                #     mod_slot[k] += 3
                # except:
                #     print(k)
    #         print(mod_slot[int(start_freq / 5 * Slen): int((start_freq + bandwidth) / 5 * Slen)])
            out_mod_slot = ''
            for k in range(len(mod_slot)):
                out_mod_slot = out_mod_slot + str(mod_slot[k]) + ' '
            oid.write(out_mod_slot + '\n')
    oid.close()
    print(out_filename + ' is generated')


for filename in glob.glob(ori_folder + '*.txt'):
    add_spec(filename, start_freq, bandwidth, period)