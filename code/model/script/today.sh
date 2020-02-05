#!/bin/bash
cd ~/Anomaly-Detection-for-Spectrum/code/model/

python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_ab_train_LOS-5M-USRP2 1250 128 0
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_ab_train_LOS-5M-USRP3 1250 128 0
python evaluation_joblib.py 10 1000 250 ryerson_train ryerson_ab_train_Dynamics-5M-USRP1 1250 128 0
