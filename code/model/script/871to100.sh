cd ~/Anomaly-Detection-for-Spectrum/code/model

python featurization.py 10 100 25 871_ab_LOS-5M-USRP2 abnormal 4
python featurization.py 10 100 25 871_ab_LOS-5M-USRP2 abnormal 4
python featurization.py 10 100 25 871_ab_LOS-5M-USRP3 abnormal 4
python featurization.py 10 100 25 871_ab_NLOS-5M-USRP1 abnormal 4
python featurization.py 10 100 25 871_ab_Dynamics-5M-USRP1 abnormal 4

cd /net/adv_spectrum/
chmod -R 777 .
