#!/bin/bash
cd ..

python featurization.py 10 10 5 ryerson_train normal 20
python featurization.py 10 10 5 ryerson_ab_train_LOS-5M-USRP1 abnormal 20
python featurization.py 10 10 5 ryerson_ab_train_LOS-5M-USRP2 abnormal 20
python featurization.py 10 10 5 ryerson_ab_train_LOS-5M-USRP3 abnormal 20
python featurization.py 10 10 5 ryerson_ab_train_NLOS-5M-USRP1 abnormal 20
python featurization.py 10 10 5 ryerson_ab_train_Dynamics-5M-USRP1 abnormal 20
python training.py 10 10 5 ryerson_train 5 15 256 300 1

python evaluation_joblib.py 10 10 5 ryerson_train ryerson_ab_train_LOS-5M-USRP1 15 256 1
python evaluation_joblib.py 10 10 5 ryerson_train ryerson_ab_train_LOS-5M-USRP2 15 256 1
python evaluation_joblib.py 10 10 5 ryerson_train ryerson_ab_train_LOS-5M-USRP3 15 256 1
python evaluation_joblib.py 10 10 5 ryerson_train ryerson_ab_train_NLOS-5M-USRP1 15 256 1
python evaluation_joblib.py 10 10 5 ryerson_train ryerson_ab_train_Dynamics-5M-USRP1 15 256 1
chmod -R 777 .
