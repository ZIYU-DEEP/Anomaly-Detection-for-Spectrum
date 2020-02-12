cd ~/Anomaly-Detection-for-Spectrum/code/model

python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP1 125 128 2
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP2 125 128 2
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_LOS-5M-USRP3 125 128 2
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_NLOS-5M-USRP1 125 128 2
python evaluation_joblib.py 10 100 25 ryerson_train ryerson_ab_train_Dynamics-5M-USRP1 125 128 2
