cd ../../../preprocessing
python add_raw.py 6 LOS-5M-10M--100dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-10M--100dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-10M--100dBm-usrp2 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp2 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp2
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-10M--100dBm-usrp2 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-10M--100dBm-usrp3 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp3 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--100dBm-usrp3
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-10M--100dBm-usrp3 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-10M--90dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--90dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--90dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-10M--90dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-10M--80dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--80dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-10M--80dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-10M--80dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 NLOS-5M-10M--100dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_NLOS-5M-10M--100dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_NLOS-5M-10M--100dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_NLOS-5M-10M--100dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 Dynamics-5M-10M--100dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_Dynamics-5M-10M--100dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_Dynamics-5M-10M--100dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_Dynamics-5M-10M--100dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-20M--100dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-20M--100dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-20M--100dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-20M--100dBm-usrp1 abnormal 6

cd ../preprocessing
python add_raw.py 6 LOS-5M-IDLE--100dBm-usrp1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-IDLE--100dBm-usrp1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-IDLE--100dBm-usrp1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-IDLE--100dBm-usrp1 abnormal 6
