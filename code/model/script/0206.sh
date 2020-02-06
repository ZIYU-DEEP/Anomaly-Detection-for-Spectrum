#!/bin/bash
cd ~/Anomaly-Detection-for-Spectrum/code/model

python featurization.py 10 100 25 ryerson_ab_train_LOS-5M-USRP1 8
python featurization.py 10 100 25 ryerson_ab_train_LOS-5M-USRP2 8
python featurization.py 10 100 25 ryerson_ab_train_LOS-5M-USRP3 8
python featurization.py 10 100 25 ryerson_ab_train_NLOS-5M-USRP1 8
python featurization.py 10 100 25 ryerson_ab_train_Dynamics-5M-USRP1 8
chmod -R 777 .

python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP1 125 128 1
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP2 125 128 1
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP3 125 128 1
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_NLOS-5M-USRP1 125 128 1
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_Dynamics-5M-USRP1 125 128 1
chmod -R 777 .
