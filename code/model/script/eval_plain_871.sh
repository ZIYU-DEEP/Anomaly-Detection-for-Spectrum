cd ~/Anomaly-Detection-for-Spectrum/code/model

python evaluation_joblib.py 10 1000 250 871 LOS-5M-USRP2 1250 128 0
python evaluation_joblib.py 10 1000 250 871 LOS-5M-USRP3 1250 128 0
python evaluation_joblib.py 10 1000 250 871 NLOS-5M-USRP1 1250 128 0
python evaluation_joblib.py 10 1000 250 871 Dynamics-5M-USRP1 1250 128 0

python evaluation_joblib.py 10 100 25 871 LOS-5M-USRP2 125 128 0
python evaluation_joblib.py 10 100 25 871 LOS-5M-USRP3 125 128 0
python evaluation_joblib.py 10 100 25 871 NLOS-5M-USRP1 125 128 0
python evaluation_joblib.py 10 1000 25 871 Dynamics-5M-USRP1 125 128 0

cd /net/adv_spectrum/
chmod -R 777 .
