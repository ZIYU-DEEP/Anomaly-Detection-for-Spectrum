  
cd ~/Anomaly-Detection-for-Spectrum/code/model

python evaluation_joblib.py 10 1000 250 ryerson_train LOS-5M-USRP2 1250 128 3

python evaluation_joblib.py 10 1000 250 ryerson_train LOS-5M-USRP3 1250 128 3

python evaluation_joblib.py 10 1000 250 ryerson_train NLOS-5M-USRP1 1250 128 3

python evaluation_joblib.py 10 1000 250 ryerson_train Dynamics-5M-USRP1 1250 128 3

cd /net/adv_spectrum/
chmod -R 777 .
