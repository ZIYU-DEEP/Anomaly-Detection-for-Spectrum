cd ../../../preprocessing

python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-10M--100dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-10M--100dBm-usrp2 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-10M--100dBm-usrp3 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-10M--90dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-10M--80dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/NLOS-5M-10M--100dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/Dynamics-5M-10M--100dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-20M--100dBm-usrp1 10 6
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/LOS-5M-IDLE--100dBm-usrp1 10 6

cd ../model
python featurization.py 10 1000 250 LOS-5M-10M--100dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 LOS-5M-10M--100dBm-usrp2 abnormal 6
python featurization.py 10 1000 250 LOS-5M-10M--100dBm-usrp3 abnormal 6
python featurization.py 10 1000 250 LOS-5M-10M--90dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 LOS-5M-10M--80dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 NLOS-5M-10M--100dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 Dynamics-5M-10M--100dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 LOS-5M-20M--100dBm-usrp1 abnormal 6
python featurization.py 10 1000 250 LOS-5M-IDLE--100dBm-usrp1 abnormal 6