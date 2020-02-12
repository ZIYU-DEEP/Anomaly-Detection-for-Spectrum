cd ~/Anomaly-Detection-for-Spectrum/code/model

python featurization.py 10 100 25 LOS-5M-USRP2 abnormal 4
python evaluation_joblib.py 10 100 25 ryerson_train LOS-5M-USRP2 125 128 3

python featurization.py 10 100 25 LOS-5M-USRP3 abnormal 4
python evaluation_joblib.py 10 100 25 ryerson_train LOS-5M-USRP3 125 128 3

python featurization.py 10 100 25 NLOS-5M-USRP1 abnormal 4
python evaluation_joblib.py 10 100 25 ryerson_train NLOS-5M-USRP1 125 128 3

python featurization.py 10 100 25 Dynamics-5M-USRP1 abnormal 4
python evaluation_joblib.py 10 100 25 ryerson_train Dynamics-5M-USRP1 125 128 3

cd /net/
chmod -R 777 .
