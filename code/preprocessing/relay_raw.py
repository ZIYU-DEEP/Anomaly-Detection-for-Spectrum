import numpy as np
import os
import sys
import glob
import control
import multiprocessing as mp
import random

core = int(sys.argv[1])
power_level = int(sys.argv[2])  ## power dB
file_out = '.dat'  ## name seq: real_BS-fake_BS-power_level
time_interval = 1  ## seconds
all_time = 40  ## seconds
step_sec = 10000000  ## step number per sec, i.e. sample rate
mag_level = control.db2mag(power_level)  ## power magnitude

def print_raw(file):
    ## print out raw data of a block
    fid = open(file, 'rb')
    num = np.fromfile(fid, np.float32, count=400000000)
    print(num, np.shape(num))
    return num


def add_relay_raw(real_BS, file_out):
    rid = open(real_BS, 'rb')
    oid = open(file_out, 'a')
    for i in range(int(all_time/time_interval)):
        num = np.fromfile(rid, np.float32, count=time_interval * step_sec)
        if i % 2 == 0:
            num.tofile(oid)
        else:
            addnum = num * (1 + mag_level)
            addnum.tofile(oid)


real_BS_path = '/net/adv_spectrum/data/raw/normal/ryerson_ab_train'
ry_t1_path = '/net/adv_spectrum/data/raw/normal/ryerson_t1'
ry_t2_path = '/net/adv_spectrum/data/raw/normal/ryerson_t2'
sr_path = '/net/adv_spectrum/data/raw/normal/searle'
dt_path = '/net/adv_spectrum/data/raw/normal/downtown'
jcl_path = '/net/adv_spectrum/data/raw/normal/JCL'
abnormal_path = '/net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_' + str(power_level) + 'dB/'
normal_path = '/net/adv_spectrum/data/raw/normal/'

real_BS_files = glob.glob(real_BS_path + '/*.dat')
real_per_core = 23 // core


if not os.path.exists(abnormal_path):
    os.mkdir(abnormal_path)
    print(abnormal_path + ' Created')
    

def add_relay_raw_batch(index):
    files = [real_BS_files[i*core + index] for i in range(real_per_core)]
    # print(files)
    for file in files:
        print('start relaying ' + file)
        file_out = file.replace('.dat', '_relay.dat').split('/')[-1]
        if not os.path.exists(file_out):
            print(file_out)
            add_relay_raw(file, abnormal_path + file_out)
            print(file_out + ' is successfully generated')
        else:
            print(file_out + ' already exists, start processing the next file.')


def multicore_add_relay_raw_batch(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(add_relay_raw_batch, index)
	return

if __name__ == "__main__":
    multicore_add_relay_raw_batch(core, range(core))