import numpy as np
import os
import sys
import glob
import control
import multiprocessing as mp
import random

core = int(sys.argv[1])
fake_BS = sys.argv[2]
real_num = int(sys.argv[3])
fake_num = int(sys.argv[4])
# real_BS = '.dat'
# fake_BS = '.dat'
file_out = '.dat'  ## name seq: real_BS-fake_BS-power_level
time_interval = 5  ## seconds
all_time = 40  ## seconds
step_sec = 10000000  ## step number per sec, i.e. sample rate
power_level = 3  ## power dB
# in_path = '/net/adv_spectrum/data/raw/normal/ryerson2/'
# ry_in_path = '/net/adv_spectrum/data/raw/normal/ryerson2/'
# jcl_in_path = '/net/adv_spectrum/data/raw/normal/JCL/'


def print_raw(file):
    ## print out raw data of a block
    fid = open(file, 'rb')
    num = np.fromfile(fid, np.float32, count=400000000)
    print(num, np.shape(num))
    return num


def add_raw(real_BS, fake_BS, file_out):
    rid = open(real_BS, 'rb')
    fid = open(fake_BS, 'rb')
    oid = open(file_out, 'a')
    rnum = np.fromfile(rid, np.float32, count=400000000)
    fnum = np.fromfile(fid, np.float32, count=400000000)
    addnum = rnum + fnum
    addnum.tofile(oid)


def add_raw_time_shift(real_BS, file_out, shift_step):
    stps = 1000 * 128 * 10 * 2 #shift_time per step
    rid = open(real_BS, 'rb')
    oid = open(file_out, 'a')
    rnum = np.fromfile(rid, np.float32, count=400000000)
    # print(rnum, np.shape(rnum))
    rnum2 = rnum[: -stps * shift_step]
    # print(rnum2, np.shape(rnum2))
    srnum = rnum[stps * shift_step:]
    # print(srnum, np.shape(srnum))
    addnum = srnum + rnum2
    # print(addnum, np.shape(addnum))
    addnum.tofile(oid)


def add_raw_time_shift_batch(real_BS_path, abnormal_path, shift_step):
    if not os.path.exists(abnormal_path):
        os.mkdir(abnormal_path)
        print(abnormal_path + ' Created')
    for file in glob.glob(real_BS_path + '/*.dat'):
        print(file)
        print('start adding ' + file + ' and ' + str(shift_step) + '-second-shifted file')
        file_out = file.split('/')[-1].split('.')[0] + '_ts' + str(shift_step) + '.dat'
        print(file_out)
        add_raw_time_shift(file, abnormal_path + file_out, shift_step)


def add_same_raw(real_BS, fake_BS, file_out, time_interval):
    ## add files on the same raw segment
    rid = open(real_BS, 'rb')
    fid = open(fake_BS, 'rb')
    oid = open(file_out, 'a')
    trash_count = 102400
    ftrash = np.fromfile(fid, np.float32, count=trash_count)
    fnum = np.fromfile(fid, np.float32, count=time_interval * step_sec)
    for i in range(int(all_time / time_interval)):
        rnum = np.fromfile(rid, np.float32, count=time_interval * step_sec)
        if i % 2 == 0:
            rnum.tofile(oid)
        else:
            addnum = rnum + fnum
            addnum.tofile(oid)


def add_diff_raw(real_BS, fake_BS, file_out, time_interval, power_level):
    ## add files on the different raw segments，with different power level
    rid = open(real_BS, 'rb')
    fid = open(fake_BS, 'rb')
    oid = open(file_out, 'a')
    mag_level = control.db2mag(power_level)  ## power magnitude
    for i in range(int(all_time / time_interval)):
        rnum = np.fromfile(rid, np.float32, count=time_interval * step_sec)
        fnum = np.fromfile(fid, np.float32, count=time_interval * step_sec) * mag_level
        if i % 2 == 0:
            rnum.tofile(oid)
        else:
            addnum = rnum + fnum
            addnum.tofile(oid)


def add_same_batch(path, fake_BS):
    ## add same FBS signal on a batch series
    if not os.path.exists(path):
        os.mkdir(path)
        print(path + ' Created')
    for file in glob.glob(in_path + '*.dat'):
        print(file)
        print('start adding ' + file + ' and ' + fake_BS)
        if file != fake_BS:
            file_out = file.split('/')[-1].split('_')[0] + '_' + fake_BS
            print(file_out)
            add_same_raw(file, in_path + fake_BS, path + file_out, time_interval)


def add_diff_batch(path, real_BS, power_level):
    ## add same FBS signal on a batch series
    G_path = path + '_G' + str(power_level) + '/'
    if not os.path.exists(G_path):
        os.mkdir(G_path)
        print(G_path + ' Created')
    for file in glob.glob(in_path + '*.dat'):
        if file != real_BS:
            file_out = real_BS.split('/')[-1].split('_')[0] + '_' + file.split('/')[-1]
            print('start adding ' + file + ' and ' + real_BS +', on power level ' + str(power_level))
            add_diff_raw(in_path + real_BS, file, G_path + file_out, time_interval, power_level)


#add_same_batch('/net/adv_spectrum/data/raw/abnormal/ryerson2_same/', '1518560024_880M_5m.dat')
#for i in range(10):
#    add_diff_batch('/net/adv_spectrum/data/raw/abnormal/ryerson2_diff', '1518560024_880M_5m.dat', i*3 - 3)
# for i in range(3):
#     G_path = '/net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix' + '_G' + str(i*3) + '/'
#     if not os.path.exists(G_path):
#         os.mkdir(G_path)
#         print(G_path + ' Created')
#     real_BS = ry_in_path + '1518560024_880M_5m.dat'
#     fake_BS = jcl_in_path + '1572728951_880M_5m.dat'
#     file_out = real_BS.split('/')[-1].split('_')[0] + '_' + fake_BS.split('/')[-1]
#     print('start adding ' + real_BS + ' and ' + fake_BS + ', on power level ' + str(i*3))
#     add_diff_raw(real_BS, fake_BS, G_path + file_out, 5, i*3)

real_BS_path = '/net/adv_spectrum/data/raw/normal/downtown'
ry_t1_path = '/net/adv_spectrum/data/raw/normal/ryerson_t1'
ry_t2_path = '/net/adv_spectrum/data/raw/normal/ryerson_t2'
sr_path = '/net/adv_spectrum/data/raw/normal/searle'
dt_path = '/net/adv_spectrum/data/raw/normal/downtown'
jcl_path = '/net/adv_spectrum/data/raw/normal/JCL'
abnormal_path = '/net/adv_spectrum/data/raw/abnormal/downtown'
FBSpath = '/net/adv_spectrum/data/raw/abnormal/'

# add_raw_batch(real_BS_path, ry_t1_path, abnormal_path + '_' + ry_t1_path.split('/')[-1] + '/')

print(fake_BS)
fake_BS_path = FBSpath + fake_BS
print(fake_BS_path)
abnormal_path = abnormal_path + '_' + fake_BS_path.split('/')[-1] + '/'
real_BS_files = glob.glob(real_BS_path + '/*.dat')
print(np.shape(real_BS_files))
rand_list = random.sample(range(10), real_num)
print(rand_list)
real_BS_files = [real_BS_files[i] for i in rand_list]
fake_BS_files = glob.glob(fake_BS_path + '/*.dat')
print(np.shape(fake_BS_files))
real_per_core = real_num // core

if not os.path.exists(abnormal_path):
    os.mkdir(abnormal_path)
    print(abnormal_path + ' Created')
    
def add_raw_batch(index):
    files = [real_BS_files[i*core + index] for i in range(real_per_core)]
    # print(files)
    for file in files:
        rand_list = random.sample(range(5), fake_num) 
        # print(rand_list)
        for rand in rand_list:
            fake_BS = fake_BS_files[rand]
            # print(file)
            print('start adding ' + file + ' and ' + str(index) + ' ' + fake_BS)
            file_out = file.split('/')[-1].split('_')[0] + '_' + fake_BS.split('/')[-1]
            if not os.path.exists(file_out):
                print(file_out)
                add_raw(file, fake_BS, abnormal_path + file_out)
                print(file_out + ' is successfully generated')
            else:
                print(file_out + ' already exists, start processing the next file.')

# for path in [ry_t2_path, sr_path, dt_path, jcl_path]:
#      add_raw_batch(real_BS_path, path, abnormal_path + '_' + path.split('/')[-1] + '/')

# abnormal_path = '/net/adv_spectrum/data/raw/abnormal/ryerson_test'

# for num in range(10):
#     add_raw_time_shift_batch(real_BS_path, abnormal_path + '_ts' + str(2*num + 1) + '/', 2*num+1)

def multicore_add_raw_batch(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(add_raw_batch, index)
	return

if __name__ == "__main__":
    multicore_add_raw_batch(core, range(core))