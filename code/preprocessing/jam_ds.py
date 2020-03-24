import numpy as np
import sys
import os
import glob
import multiprocessing as mp

ori_folder = '/net/adv_spectrum/data/downsample/downsample_10/normal/' + sys.argv[1] + '/'
so_folder = '/net/adv_spectrum/data/downsample/downsample_10/abnormal/' + sys.argv[1] + '_' + sys.argv[4] + '/'
if not os.path.exists(so_folder):
    os.makedirs(so_folder)
    print(so_folder + ' is created')
start_freq = float(sys.argv[2])
bandwidth = float(sys.argv[3])
mode = sys.argv[4]
core = int(sys.argv[5])

def bits_to_spec(Sig, size):
    Y = np.fft.fft(Sig, 2560)
    P1 = np.fft.fftshift(Y)
    P2 = np.absolute(P1)
    amp = 10*np.log10(P2/2560)
    spec = amp[::int(2560/size)]
    return spec

files = glob.glob(ori_folder + '*.txt')
file_num = np.shape(files)[0]
file_per_core = file_num // core
print(file_per_core)

def add_spec(index):
    ## Input: filename of the downsample version of Signals 
    ## Start_freq: the location where the freq is started in MHz
    ## bandwidth: channel bandwidth, in MHz
    ## period: the signal injection period, in ms
    
    time_step = 1000
    Slen = 128
    start_point = 0 # in lines, this is the startline (in time domain)
    ## configure the info to be transmitted first
    fs = 44000  # sampling rate
    baud = 300  # symbol rate
    Nbits = 10  # number of bits
    f0 = 880000000
    Ns = int(fs/baud)
    N = Nbits * Ns

    filenames = [files[i*core + index] for i in range(file_per_core)]
    for filename in filenames:
        line_count = 0
        fid = open(filename,'r')
        out_filename = so_folder + filename.split('/')[-1].replace('ap.txt', mode + '_ap.txt')
        oid = open(out_filename, 'a')
        print(out_filename + ' is being processed.')
        
        # skip the former start_point lines
        for i in range(start_point):
            fid.readline()
            line_count += 1
        
        # modify the targetted slots
        # flag = 0 # to indicate the end of file
        while(True):
            slot = fid.readline()
            if line_count == 156000:
                break

            mod_slot = [float(j) for j in slot.split()]
            # print(np.shape(mod_slot))
            if np.shape(mod_slot)[0] == 128:
                line_count += 1
                sample_size = int((start_freq + bandwidth) / 5 * Slen) - int(start_freq / 5 * Slen)
                if mode == 'wn':
                    white_noise = np.random.normal(3, 0, size=sample_size)
                    # print(mod_slot[int(start_freq / 5 * Slen): int((start_freq + bandwidth) / 5 * Slen)])
                    for k in range(int(start_freq / 5 * Slen), int((start_freq + bandwidth) / 5 * Slen)):
                        mod_slot[k] += white_noise[k - int(start_freq / 5 * Slen)]
                    out_mod_slot = ''
                    for k in range(len(mod_slot)):
                        out_mod_slot = out_mod_slot + str(mod_slot[k]) + ' '
                    oid.write(out_mod_slot + '\n')
                else:
                    ## configure the info to be transmitted first
                    bits = np.random.randn(Nbits,1) > 0
                    if mode == 'ask':
                        M = np.tile(bits,(1,Ns))
                        t = np.r_[0.0:N]/fs
                        ask = M.ravel()*np.sin(2*np.pi*f0*t)
                        spec = bits_to_spec(ask, sample_size)
                    elif mode == 'psk':
                        M = np.tile(bits*2-1,(1,Ns))
                        t = np.r_[0.0:N]/fs
                        bpsk = M.ravel()*np.sin(2*np.pi*f0*t)
                        spec = bits_to_spec(bpsk, sample_size)
                    elif mode == 'fsk':
                        M = np.tile(bits*2-1,(1,Ns))
                        delta_f = 600
                        ph = 2*np.pi*np.cumsum(f0 + M.ravel()*delta_f)/fs
                        t = np.r_[0.0:N]/fs
                        fsk = np.sin(ph)
                        spec = bits_to_spec(fsk, sample_size)
                    elif mode == 'qam':
                        Nbits = 16  # number of bits
                        N = Nbits * Ns
                        code = np.array((-2-2j, -2-1j,-2+2j,-2+1j,-1-2j,-1-1j,-1+2j,-1+1j,+2-2j,+2-1j,+2+2j,+2+1j,1-2j,+1-1j,1+2j,1+1j))/2
                        bits = np.int16(np.random.rand(Nbits,1)*16) 
                        M = np.tile(code[bits],(1,Ns))
                        t = np.r_[0.0:N]/fs
                        qam = np.real(M.ravel()*np.exp(1j*2*np.pi*f0*t))/np.sqrt(2)/2
                        spec = bits_to_spec(qam, sample_size)

                    adjust = 3 - np.mean(spec)
                    for k in range(int(start_freq / 5 * Slen), int((start_freq + bandwidth) / 5 * Slen)):
                        mod_slot[k] += spec[k - int(start_freq / 5 * Slen)] + adjust
                    out_mod_slot = ''
                    for k in range(len(mod_slot)):
                        out_mod_slot = out_mod_slot + str(mod_slot[k]) + ' '
                    oid.write(out_mod_slot + '\n')

        oid.close()
        print(out_filename + ' is generated')
        break

def multicore_add_spec(core, index):
	## run faster using multiprocessing on multicores of CPU
	pool = mp.Pool(processes = core)
	pool.map(add_spec, index)
	return

if __name__ == "__main__":
	# multicore run downsampling
    multicore_add_spec(core, range(core))