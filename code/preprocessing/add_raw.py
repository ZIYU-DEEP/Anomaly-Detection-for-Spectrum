import numpy as np
import os
import glob
import control

real_BS = '.dat'
fake_BS = '.dat'
file_out = '.dat'  ## name seq: real_BS-fake_BS-power_level
time_interval = 5  ## seconds
all_time = 40  ## seconds
step_sec = 10000000  ## step number per sec, i.e. sample rate
power_level = 3  ## power dB
in_path = '/net/adv_spectrum/data/raw/normal/ryerson2/'
ry_in_path = '/net/adv_spectrum/data/raw/normal/ryerson2/'
jcl_in_path = '/net/adv_spectrum/data/raw/normal/JCL/'


def print_raw(file):
    ## print out raw data of a block
    fid = open(file, 'rb')
    num = np.fromfile(fid, np.float32, count=400000000)
    print(num, np.shape(num))
    return num


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
for i in range(10):
    G_path = '/net/adv_spectrum/data/raw/abnormal/ry2_jcl_mix' + '_G' + str(i*3 -3) + '/'
    if not os.path.exists(G_path):
        os.mkdir(G_path)
        print(G_path + ' Created')
    real_BS = ry_in_path + '1518560024_880M_5m.dat'
    fake_BS = jcl_in_path + '1572728951_880M_5m.dat'
    file_out = real_BS.split('/')[-1].split('_')[0] + '_' + fake_BS.split('/')[-1]
    add_diff_raw(real_BS, fake_BS, G_path + file_out, 5, i*3 - 3)