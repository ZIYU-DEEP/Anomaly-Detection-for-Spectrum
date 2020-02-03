cd ../../../preprocessing
python add_raw.py 6 LOS-5M-USRP1 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP1 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP1
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-USRP1 abnormal 6
chmod 777 -R /net/adv_spectrum

cd ../preprocessing
python add_raw.py 6 LOS-5M-USRP2 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP2 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP2
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-USRP2 abnormal 6
chmod 777 -R /net/adv_spectrum

cd ../preprocessing
python add_raw.py 6 LOS-5M-USRP3 12 5
python downsample_parse_data.py /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP3 10 6
rm -rf /net/adv_spectrum/data/raw/abnormal/ryerson_ab_train_LOS-5M-USRP3
cd ../model
python featurization.py 10 1000 250 ryerson_ab_train_LOS-5M-USRP3 abnormal 6
chmod 777 -R /net/adv_spectrum